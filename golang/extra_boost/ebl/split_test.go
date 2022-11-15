package extra_boost_lib

import (
	"encoding/json"
	"fmt"
	"gonum.org/v1/gonum/mat"
	"os"
	"path"
	"testing"
)

func CreateTestEMatrix() (EMatrix, int) {
	h := 10
	extraW := 3

	AllWeights := []*mat.Dense{mat.NewDense(extraW, 1, []float64{7, 11, 13}), mat.NewDense(extraW, 1, []float64{1979, 18, 25})}
	nWeights := len(AllWeights)

	rawTarget := make([]float64, nWeights*h)
	target := mat.NewDense(h*nWeights, 1, rawTarget)

	parabolaArray := make([]float64, nWeights*h*extraW)

	rangeArray := make([]float64, h*nWeights)
	featuresInter := mat.NewDense(nWeights*h, 1, rangeArray)

	recordIds := make([]int, 0)
	globalInd := 0

	for weightIndex, currentWeight := range AllWeights {
		for ind := 0; ind < h; ind++ {
			rangeArray[weightIndex*h+ind] = float64(weightIndex*h + ind + 1)
		}

		timeAperture := 1.0
		for p := 0; p < h; p++ {
			t := float64(p) * timeAperture / (float64(h) - 1.0)

			parabolaArray[weightIndex*h*extraW+extraW*p+0] = 1.0
			parabolaArray[weightIndex*h*extraW+extraW*p+1] = t
			parabolaArray[weightIndex*h*extraW+extraW*p+2] = t * t

			currentTarget := 0.0
			for pp := 0; pp < extraW; pp++ {
				currentTarget += parabolaArray[weightIndex*h*extraW+extraW*p+pp] * currentWeight.At(pp, 0)
			}
			target.Set(h*weightIndex+p, 0, currentTarget)
			recordIds = append(recordIds, globalInd)
			globalInd++
		}
	}
	featuresExtra := mat.NewDense(nWeights*h, extraW, parabolaArray)

	//fmt.Println("FeaturesInter")
	//fmt.Printf("%.4g\n", mat.Formatted(FeaturesInter))
	//fmt.Println("FeaturesExtra")
	//fmt.Printf("%.4g\n", mat.Formatted(FeaturesExtra))
	//fmt.Println("Target")
	//fmt.Printf("%.4g\n", mat.Formatted(Target))

	return EMatrix{FeaturesInter: featuresInter, FeaturesExtra: featuresExtra, Target: target, RecordIds: recordIds}, nWeights
}

func TestScanForSplit(t *testing.T) {
	testEMatrix, nWeights := CreateTestEMatrix()

	h, _, d := testEMatrix.validatedDimensions()
	rawHessian := testEMatrix.allocateArrays()

	bias := mat.NewDense(nWeights*h, 1, nil)

	bestSplit := scanForSplitCluster(testEMatrix, h, d, 0, bias, MseLoss{}, 1e-6, rawHessian, 0)

	fmt.Println("delta up:")
	fmt.Printf("%.4g\n", mat.Formatted(bestSplit.deltaUp))

	fmt.Println("delta down:")
	fmt.Printf("%.4g\n", mat.Formatted(bestSplit.deltaDown))

	fmt.Println("delta current:")
	fmt.Printf("%.4g\n", mat.Formatted(bestSplit.deltaCurrent))
}

func CreateTestEMatrixWithClusters() EMatrix {
	FeaturesInter, FeaturesExtra, Target, RecordIds := GenerateDebugData()

	return EMatrix{
		FeaturesInter: FeaturesInter,
		FeaturesExtra: FeaturesExtra,
		Target:        Target,
		RecordIds:     RecordIds,
	}
}

func TestScanForSplitWithClusters(t *testing.T) {
	testEMatrix := CreateTestEMatrixWithClusters()

	h, _, d := testEMatrix.validatedDimensions()
	rawHessian := testEMatrix.allocateArrays()

	bias := mat.NewDense(h, 1, nil)

	bestSplit := scanForSplitCluster(testEMatrix, h, d, 0, bias, MseLoss{}, 1e-6, rawHessian, 0)

	fmt.Println("delta up:")
	fmt.Printf("%.4g\n", mat.Formatted(bestSplit.deltaUp))

	fmt.Println("delta down:")
	fmt.Printf("%.4g\n", mat.Formatted(bestSplit.deltaDown))

	fmt.Println("delta current:")
	fmt.Printf("%.4g\n", mat.Formatted(bestSplit.deltaCurrent))
}

func TestArgSort(t *testing.T) {
	dataPath := "/home/tass/database/app_in_the_air/demand_predictions/current_data_set/"
	flnmInterTrainDebug := "inter_train_debug" // "tiny_array"
	ext := ".npy"

	pathInterTrain := path.Join(dataPath, flnmInterTrainDebug+ext)
	interTrain := ReadNpy(pathInterTrain)

	featureAs := columnArgsort(interTrain.ColView(0))

	for ind := 0; ind < 100; ind++ {
		fmt.Println(featureAs[ind], interTrain.At(featureAs[ind], 0))
	}

	flnmAS := "tiny_feature_as"
	pathAS := path.Join(dataPath, flnmAS)

	dst, err := os.Create(pathAS)
	HandleError(err)
	defer func() { HandleError(dst.Close()) }()

	jsonBytes, err := json.MarshalIndent(featureAs, "", "  ")
	HandleError(err)

	dst.Write(jsonBytes)
}

func TestArgSortTiny(t *testing.T) {
	f := mat.NewDense(5, 1, []float64{5.0, 4.0, 6.0, 1.0, 2.0})

	fAs := columnArgsort(f.ColView(0))

	if fAs[0] != 3 || fAs[1] != 4 || fAs[2] != 1 || fAs[3] != 0 || fAs[4] != 2 {
		t.Errorf("wrong argsort %v", fAs)
	}
}

func TestScanForSplit59(t *testing.T) {
	dataPath := "/home/tass/database/app_in_the_air/demand_predictions/current_data_set/"
	flnmInter := "chunk_59_inter"
	flnmExtra := "chunk_59_extra"
	flnmTarget := "chunk_59_target"
	ext := ".npy"

	pathInter := path.Join(dataPath, flnmInter+ext)
	pathExtra := path.Join(dataPath, flnmExtra+ext)
	pathTarget := path.Join(dataPath, flnmTarget+ext)

	inter := ReadNpy(pathInter)
	extra := ReadNpy(pathExtra)
	target := ReadNpy(pathTarget)

	ematrix := EMatrix{FeaturesInter: inter, FeaturesExtra: extra, Target: target}

	h, _, d := ematrix.validatedDimensions()
	rawHessian := ematrix.allocateArrays()

	bias := mat.NewDense(h, 1, nil)

	bestSplit := scanForSplitCluster(ematrix, h, d, 0, bias, MseLoss{}, 1e-6, rawHessian, 0)

	fmt.Println("delta up:")
	fmt.Printf("%.4g\n", mat.Formatted(bestSplit.deltaUp))

	fmt.Println("delta down:")
	fmt.Printf("%.4g\n", mat.Formatted(bestSplit.deltaDown))

	fmt.Println("delta current:")
	fmt.Printf("%.4g\n", mat.Formatted(bestSplit.deltaCurrent))
}
