package ebl

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestEBoosterHandlesConstantFeatures(t *testing.T) {
	rows := 128
	featuresInter := mat.NewDense(rows, 3, nil)
	featuresExtra := mat.NewDense(rows, 4, nil)
	target := mat.NewDense(rows, 1, nil)

	for i := 0; i < rows; i++ {
		tVal := float64(i) / float64(rows-1)
		featuresExtra.Set(i, 0, 1.0)
		featuresExtra.Set(i, 1, tVal)
		featuresExtra.Set(i, 2, math.Sin(50*tVal))
		featuresExtra.Set(i, 3, math.Cos(50*tVal))
		val := 0.3 + 0.5*tVal + 0.2*math.Sin(50*tVal) - 0.1*math.Cos(50*tVal)
		target.Set(i, 0, val)
	}

	em := EMatrix{
		FeaturesInter: featuresInter,
		FeaturesExtra: featuresExtra,
		Target:        target,
		RecordIds:     make([]int, rows),
	}
	for i := range em.RecordIds {
		em.RecordIds[i] = i
	}

	params := EBoosterParams{
		Matrix:       em,
		NStages:      1,
		RegLambda:    1e-6,
		MaxDepth:     3,
		LearningRate: 1.0,
		LossKind:     MseLoss{},
		ThreadsNum:   1,
	}

	booster := NewEBooster(params)
	if len(booster.Trees) != 1 {
		t.Fatalf("expected 1 tree, got %d", len(booster.Trees))
	}
	tree := booster.Trees[0]
	if len(tree.TreeNodes) != 1 {
		t.Fatalf("expected single node tree, got %d nodes", len(tree.TreeNodes))
	}
	node := tree.TreeNodes[0]
	if !node.NoSplit {
		t.Fatalf("expected node to be marked as NoSplit")
	}
	if node.FeatureNumber != -1 {
		t.Fatalf("expected FeatureNumber -1, got %d", node.FeatureNumber)
	}

	prediction := booster.PredictValue(featuresInter, featuresExtra, nil)
	sumSq := 0.0
	for i := 0; i < rows; i++ {
		diff := target.At(i, 0) - prediction.At(i, 0)
		sumSq += diff * diff
	}
	rmse := math.Sqrt(sumSq / float64(rows))
	if rmse > 1e-6 {
		t.Fatalf("unexpected RMSE: %g", rmse)
	}
}
