package ebl

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"testing"
)

//It it a good place to debug arithmetics of the model
func TestRunAlgo(t *testing.T) {
	FeaturesInter := mat.NewDense(12, 1, []float64{
		1,
		1,
		1,
		1,
		1,
		1,
		3,
		3,
		3,
		3,
		3,
		3,
	})
	FeaturesExtra := mat.NewDense(12, 3, []float64{
		1.00, 0.00, 0.00,
		1.00, 0.20, 0.04,
		1.00, 0.40, 0.16,
		1.00, 0.60, 0.36,
		1.00, 0.80, 0.64,
		1.00, 1.00, 1.00,
		1.00, 0.00, 0.00,
		1.00, 0.20, 0.04,
		1.00, 0.40, 0.16,
		1.00, 0.60, 0.36,
		1.00, 0.80, 0.64,
		1.00, 1.00, 1.00,
	})
	Target := mat.NewDense(12, 1, []float64{
		1.00,
		1.52,
		2.28,
		3.28,
		4.52,
		6.00,
		10.00,
		10.02,
		9.88,
		9.58,
		9.12,
		8.50,
	})
	RecordIds := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

	bias := mat.NewDense(12, 1, []float64{
		4.1, 4.9, 5.5, 6.0, 6.5, 7.0, 4.1, 4.9, 5.5, 6.0, 6.5, 7.0,
	})

	ematrix := EMatrix{
		FeaturesInter: FeaturesInter,
		FeaturesExtra: FeaturesExtra,
		Target:        Target,
		RecordIds:     RecordIds,
	}
	clf := NewEBooster(EBoosterParams{
		Matrix:         ematrix,
		NStages:        10,
		RegLambda:      1e-6,
		MaxDepth:       2,
		LearningRate:   0.2,
		LossKind:       MseLoss{},
		PrintMessages:  []EMatrix{},
		ThreadsNum:     1,
		UnbalancedLoss: 0,
		Bias:           bias,
	})

	_ = clf

	//wd, err := os.Getwd()
	//HandleError(err)
	//fmt.Print("current working dir = ", wd)
	//clf.RenderTrees("small_example", "png", "/home/tass/database/app_in_the_air/demand_predictions/small_example/")
}

func GenerateDebugData() (FeaturesInter, FeaturesExtra, Target *mat.Dense, RecordIds []int) {
	FeaturesInter = mat.NewDense(18, 1, []float64{
		1,
		1,
		1,
		1,
		1,
		1,
		3,
		3,
		3,
		3,
		3,
		3,
		5,
		5,
		5,
		5,
		5,
		5,
	})

	FeaturesExtra = mat.NewDense(18, 3, []float64{
		1.00, 0.00, 0.00,
		1.00, 0.20, 0.04,
		1.00, 0.40, 0.16,
		1.00, 0.60, 0.36,
		1.00, 0.80, 0.64,
		1.00, 1.00, 1.00,
		1.00, 0.00, 0.00,
		1.00, 0.20, 0.04,
		1.00, 0.40, 0.16,
		1.00, 0.60, 0.36,
		1.00, 0.80, 0.64,
		1.00, 1.00, 1.00,
		1.00, 0.00, 0.00,
		1.00, 0.20, 0.04,
		1.00, 0.40, 0.16,
		1.00, 0.60, 0.36,
		1.00, 0.80, 0.64,
		1.00, 1.00, 1.00,
	})

	Target = mat.NewDense(18, 1, []float64{
		1.00,
		1.52,
		2.28,
		3.28,
		4.52,
		6.00,
		10.00,
		10.02,
		9.88,
		9.58,
		9.12,
		8.50,
		2.00,
		1.60,
		1.60,
		2.00,
		2.80,
		4.00,
	})
	RecordIds = []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}

	return
}

func GenerateDebugDataTwelve() (FeaturesInter, FeaturesExtra, Target *mat.Dense, RecordIds []int) {
	FeaturesInter = mat.NewDense(12, 1, []float64{
		1,
		1,
		1,
		1,
		1,
		1,
		3,
		3,
		3,
		3,
		3,
		3,
	})

	FeaturesExtra = mat.NewDense(12, 3, []float64{
		1.00, 0.00, 0.00,
		1.00, 0.20, 0.04,
		1.00, 0.40, 0.16,
		1.00, 0.60, 0.36,
		1.00, 0.80, 0.64,
		1.00, 1.00, 1.00,
		1.00, 0.00, 0.00,
		1.00, 0.20, 0.04,
		1.00, 0.40, 0.16,
		1.00, 0.60, 0.36,
		1.00, 0.80, 0.64,
		1.00, 1.00, 1.00,
	})

	Target = mat.NewDense(12, 1, []float64{
		1.00,
		1.52,
		2.28,
		3.28,
		4.52,
		6.00,
		10.00,
		10.02,
		9.88,
		9.58,
		9.12,
		8.50,
	})
	RecordIds = []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

	return
}

func GenerateDebugDataLinear() (FeaturesInter, FeaturesExtra, Target *mat.Dense, RecordIds []int) {
	FeaturesInter = mat.NewDense(18, 1, []float64{
		1,
		1,
		3,
		3,
		5,
		5,
		7,
		7,
		9,
		9,
		11,
		11,
		13,
		13,
		15,
		15,
		17,
		17,
	})

	FeaturesExtra = mat.NewDense(18, 2, []float64{
		1.00, 0.00,
		1.00, 1.00,
		1.00, 0.00,
		1.00, 1.00,
		1.00, 0.00,
		1.00, 1.00,
		1.00, 0.00,
		1.00, 1.00,
		1.00, 0.00,
		1.00, 1.00,
		1.00, 0.00,
		1.00, 1.00,
		1.00, 0.00,
		1.00, 1.00,
		1.00, 0.00,
		1.00, 1.00,
		1.00, 0.00,
		1.00, 1.00,
	})

	Target = mat.NewDense(18, 1, []float64{
		1.00,
		3.00,
		2.00,
		5.00,
		7.00,
		12.00,
		11.00,
		14.00,
		6.00,
		5.00,
		9.00,
		4.00,
		2.00,
		9.00,
		17.00,
		25.00,
		17.00,
		13.00,
	})
	RecordIds = []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}

	return
}

//This test is used to debug a part where thresholds are selected
//in places of interpolating features change only
func TestSplitWhereInterpolatingFeatureDiffers_00(t *testing.T) {
	FeaturesInter, FeaturesExtra, Target, RecordIds := GenerateDebugData()

	bias := mat.NewDense(18, 1, []float64{
		4.1, 4.9, 5.5, 6.0, 6.5, 7.0, 4.1, 4.9, 5.5, 6.0, 6.5, 7.0, 4.1, 4.9, 5.5, 6.0, 6.5, 7.0,
	})

	ematrix := EMatrix{
		FeaturesInter: FeaturesInter,
		FeaturesExtra: FeaturesExtra,
		Target:        Target,
		RecordIds:     RecordIds,
	}
	clf := NewEBooster(EBoosterParams{
		Matrix:         ematrix,
		NStages:        10,
		RegLambda:      1e-6,
		MaxDepth:       2,
		LearningRate:   0.2,
		LossKind:       MseLoss{},
		PrintMessages:  []EMatrix{},
		ThreadsNum:     1,
		UnbalancedLoss: 0,
		Bias:           bias,
	})

	_ = clf
	//wd, err := os.Getwd()
	//HandleError(err)
	//fmt.Print("current working dir = ", wd)
	//clf.RenderTrees("small_example", "png", "/home/tass/database/app_in_the_air/demand_predictions/small_example/")
}

//This test is used to debug a part where thresholds are selected
//in places of interpolating features change only
func TestSplitWhereInterpolatingFeatureDiffers_01(t *testing.T) {
	FeaturesInter, FeaturesExtra, Target, RecordIds := GenerateDebugDataLinear()

	bias := mat.NewDense(18, 1, []float64{
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	})

	ematrix := EMatrix{
		FeaturesInter: FeaturesInter,
		FeaturesExtra: FeaturesExtra,
		Target:        Target,
		RecordIds:     RecordIds,
	}
	h, _, d := ematrix.validatedDimensions()
	rawHessian := ematrix.allocateArrays()

	bestSplit := scanForSplitCluster(ematrix, h, d, 0, bias, MseLoss{}, 1e-6, rawHessian, 0.0)

	fmt.Println("current delta loss =", bestSplit.currentValue)
	fmt.Println("current delta weight =", bestSplit.deltaCurrent)

	fmt.Println("delta weight up =", bestSplit.deltaUp)
	fmt.Println("delta weight down =", bestSplit.deltaDown)
}

//This test is used to debug a part where thresholds are selected
//in places of interpolating features change only
func TestSplitWhereInterpolatingFeatureDiffers_02(t *testing.T) {
	FeaturesInter, FeaturesExtra, Target, RecordIds := GenerateDebugDataTwelve()

	bias := mat.NewDense(12, 1, []float64{
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	})

	ematrix := EMatrix{
		FeaturesInter: FeaturesInter,
		FeaturesExtra: FeaturesExtra,
		Target:        Target,
		RecordIds:     RecordIds,
	}
	h, _, d := ematrix.validatedDimensions()
	rawHessian := ematrix.allocateArrays()

	bestSplit := scanForSplitCluster(ematrix, h, d, 0, bias, MseLoss{}, 1e-6, rawHessian, 0.0)

	fmt.Println("current delta loss =", bestSplit.currentValue)
	fmt.Println("current delta weight =", bestSplit.deltaCurrent)

	fmt.Println("delta weight up =", bestSplit.deltaUp)
	fmt.Println("delta weight down =", bestSplit.deltaDown)
}
