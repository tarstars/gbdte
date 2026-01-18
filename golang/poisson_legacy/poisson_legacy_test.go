package poissonlegacy

import (
	"math"
	"testing"
)

func linspace(start, end float64, num int) []float64 {
	if num <= 1 {
		return []float64{start}
	}
	step := (end - start) / float64(num-1)
	values := make([]float64, num)
	for i := 0; i < num; i++ {
		values[i] = start + float64(i)*step
	}
	return values
}

func TestPoissonTinyFirstTree(t *testing.T) {
	bjids := []int{101, 102, 103, 104}
	freqs := []float64{10, 30, 10, 30}
	featuresInter := [][]float64{{1, 1}, {2, 3}, {3, 2}, {4, 4}}
	matrix, err := NewPMatrixFromIterables(bjids, freqs, featuresInter, nil, nil, nil)
	if err != nil {
		t.Fatalf("matrix: %v", err)
	}
	params := TreeBuildParams{MaxDepth: 1, LearningRate: 1, UnbalancedPenalty: 0.1, RegLambda: 1e-4}
	booster, err := Train(params, matrix, 1)
	if err != nil {
		t.Fatalf("train: %v", err)
	}
	featuresTest := [][]float64{{1, -1}, {2, 10}, {3, 0}, {4, 5}, {5, 2}}
	preds, err := booster.Predict(featuresTest, nil)
	if err != nil {
		t.Fatalf("predict: %v", err)
	}
	expected := []float64{10, 30, 10, 30, 10}
	for i := range expected {
		if math.Abs(preds[i]-expected[i]) > 1e-6 {
			t.Fatalf("pred[%d] = %.6f, want %.6f", i, preds[i], expected[i])
		}
	}
}

func TestPoissonMediumNextTree(t *testing.T) {
	bjids := []int{101, 102, 103, 104, 105, 106, 107, 108}
	freqs := []float64{27, 27, 33, 33, 87, 87, 93, 93}
	featuresInter := [][]float64{
		{1, 10},
		{1, 10},
		{1, 90},
		{1, 90},
		{9, 10},
		{9, 10},
		{9, 90},
		{9, 90},
	}
	matrix, err := NewPMatrixFromIterables(bjids, freqs, featuresInter, nil, nil, nil)
	if err != nil {
		t.Fatalf("matrix: %v", err)
	}
	params := TreeBuildParams{MaxDepth: 1, LearningRate: 0.7, UnbalancedPenalty: 0.1, RegLambda: 1e-3}
	booster, err := Train(params, matrix, 10)
	if err != nil {
		t.Fatalf("train: %v", err)
	}
	preds, err := booster.Predict(matrix.FeaturesInter, nil)
	if err != nil {
		t.Fatalf("predict: %v", err)
	}
	expected := []float64{27, 27, 33, 33, 87, 87, 93, 93}
	for i := range expected {
		if math.Abs(preds[i]-expected[i]) > 1e-3 {
			t.Fatalf("pred[%d] = %.6f, want %.6f", i, preds[i], expected[i])
		}
	}
}

func TestCountObjects(t *testing.T) {
	bjids := []int{101, 101, 101, 102, 102, 103}
	features := [][]float64{{1, 6, 4}, {2, 5, 3}, {3, 4, 5}, {4, 3, 2}, {5, 2, 1}, {6, 1, 6}}
	gax, err := makeGax(features)
	if err != nil {
		t.Fatalf("gax: %v", err)
	}
	counts := countObjects(bjids, gax)
	expected := [][]float64{
		{1, 1, 1},
		{1, 2, 1},
		{1, 2, 2},
		{2, 3, 2},
		{2, 3, 2},
		{3, 3, 3},
	}
	for i := range expected {
		for j := range expected[i] {
			if counts[i][j] != expected[i][j] {
				t.Fatalf("counts[%d][%d] = %.0f, want %.0f", i, j, counts[i][j], expected[i][j])
			}
		}
	}
}

func TestExtraPoissonFirstTree(t *testing.T) {
	bjids := []int{101, 101, 102, 102, 103, 103, 104, 104}
	freqs := []float64{10, 30, 50, 70, 30, 50, 70, 90}
	featuresInter := [][]float64{
		{1, 1, 4},
		{1, 1, 4},
		{1, 4, 1},
		{1, 4, 1},
		{3, 2, 3},
		{3, 2, 3},
		{3, 3, 2},
		{3, 3, 2},
	}
	functions := []TimeFunc{
		func(_ float64) float64 { return 1 },
		func(x float64) float64 { return x },
	}
	tRange := linspace(0, 1, 100)
	time := []float64{0.0, 0.2, 0.4, 0.6, 0.0, 0.2, 0.4, 0.6}
	matrix, err := NewPMatrixFromIterables(bjids, freqs, featuresInter, functions, tRange, time)
	if err != nil {
		t.Fatalf("matrix: %v", err)
	}
	params := TreeBuildParams{MaxDepth: 1, LearningRate: 0.2, RegLambda: 1e-8}
	booster, err := Train(params, matrix, 1)
	if err != nil {
		t.Fatalf("train: %v", err)
	}
	preds, err := booster.Predict(matrix.FeaturesInter, matrix.FeaturesExtra)
	if err != nil {
		t.Fatalf("predict: %v", err)
	}
	for i := range freqs {
		if math.Abs(preds[i]-freqs[i]) > 1e-4 {
			t.Fatalf("pred[%d] = %.6f, want %.6f", i, preds[i], freqs[i])
		}
	}
}
