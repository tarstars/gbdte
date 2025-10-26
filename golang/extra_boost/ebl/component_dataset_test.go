package ebl

import (
	"encoding/csv"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func loadComponentDataset(t *testing.T) (EMatrix, *mat.Dense, *mat.Dense, *mat.Dense) {
	t.Helper()

	path := filepath.Join("..", "..", "..", "datasets", "mse", "component_f000.csv")
	file, err := os.Open(path)
	if err != nil {
		t.Fatalf("failed to open dataset %q: %v", path, err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	rows, err := reader.ReadAll()
	if err != nil {
		t.Fatalf("failed to read csv: %v", err)
	}
	if len(rows) == 0 {
		t.Fatalf("component dataset is empty")
	}

	header := rows[0]
	colIndex := map[string]int{}
	for idx, name := range header {
		colIndex[name] = idx
	}
	reqCols := []string{"f_1", "f_2", "f_3", "e_1", "e_2", "e_3", "e_4", "y"}
	for _, c := range reqCols {
		if _, ok := colIndex[c]; !ok {
			t.Fatalf("column %q missing in csv", c)
		}
	}

	n := len(rows) - 1
	featuresInter := mat.NewDense(n, 3, nil)
	featuresExtra := mat.NewDense(n, 4, nil)
	target := mat.NewDense(n, 1, nil)

	for i := 0; i < n; i++ {
		row := rows[i+1]
		for j, name := range []string{"f_1", "f_2", "f_3"} {
			val, err := strconv.ParseFloat(row[colIndex[name]], 64)
			if err != nil {
				t.Fatalf("parse %s: %v", name, err)
			}
			featuresInter.Set(i, j, val)
		}
		for j, name := range []string{"e_1", "e_2", "e_3", "e_4"} {
			val, err := strconv.ParseFloat(row[colIndex[name]], 64)
			if err != nil {
				t.Fatalf("parse %s: %v", name, err)
			}
			featuresExtra.Set(i, j, val)
		}
		yVal, err := strconv.ParseFloat(row[colIndex["y"]], 64)
		if err != nil {
			t.Fatalf("parse y: %v", err)
		}
		target.Set(i, 0, yVal)
	}

	em := EMatrix{
		FeaturesInter: featuresInter,
		FeaturesExtra: featuresExtra,
		Target:        target,
		RecordIds:     make([]int, n),
	}
	for i := range em.RecordIds {
		em.RecordIds[i] = i
	}

	return em, featuresInter, featuresExtra, target
}

func solveCoefficients(features *mat.Dense, target *mat.Dense) []float64 {
	r, c := features.Dims()
	xt := mat.NewDense(c, r, nil)
	xt.Copy(features.T())

	xtx := mat.NewDense(c, c, nil)
	xtx.Mul(xt, features)

	xTy := mat.NewDense(c, 1, nil)
	xTy.Mul(xt, target)

	var coeff mat.Dense
	coeff.Solve(xtx, xTy)
	data := coeff.RawMatrix().Data
	out := make([]float64, len(data))
	copy(out, data)
	return out
}

func TestBoosterProducesNoSplitTreeForComponentDataset(t *testing.T) {
	em, featuresInter, featuresExtra, target := loadComponentDataset(t)

	params := EBoosterParams{
		Matrix:       em,
		NStages:      1,
		RegLambda:    1e-9,
		MaxDepth:     1,
		LearningRate: 1.0,
		LossKind:     MseLoss{},
		ThreadsNum:   1,
	}

	booster := NewEBooster(params)
	if got := len(booster.Trees); got != 1 {
		t.Fatalf("expected single tree, got %d", got)
	}
	tree := booster.Trees[0]
	if got := len(tree.TreeNodes); got != 1 {
		t.Fatalf("expected single node tree, got %d", got)
	}
	node := tree.TreeNodes[0]
	if !node.NoSplit {
		t.Fatalf("root node should be marked NoSplit")
	}
	if node.FeatureNumber != -1 {
		t.Fatalf("expected feature number -1, got %d", node.FeatureNumber)
	}

	expectedCoeffs := solveCoefficients(featuresExtra, target)
	if len(tree.LeafNodes) != 1 {
		t.Fatalf("expected single leaf, got %d", len(tree.LeafNodes))
	}
	leaf := tree.LeafNodes[0]
	if len(leaf.Prediction) != len(expectedCoeffs) {
		t.Fatalf("prediction length mismatch: %d vs %d", len(leaf.Prediction), len(expectedCoeffs))
	}

	for i := range expectedCoeffs {
		if math.Abs(leaf.Prediction[i]-expectedCoeffs[i]) > 1e-6 {
			t.Fatalf("prediction[%d]=%g, expected %g", i, leaf.Prediction[i], expectedCoeffs[i])
		}
	}

	prediction := booster.PredictValue(featuresInter, featuresExtra, nil)
	diff := mat.NewDense(prediction.RawMatrix().Rows, prediction.RawMatrix().Cols, nil)
	diff.Sub(prediction, target)
	var norm float64
	for _, v := range diff.RawMatrix().Data {
		norm += v * v
	}
	rmse := math.Sqrt(norm / float64(prediction.RawMatrix().Rows))
	if rmse > 1e-6 {
		t.Fatalf("unexpected RMSE: %g", rmse)
	}
}
