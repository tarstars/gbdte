package ebl

import (
	"github.com/sbinet/npyio"
	"gonum.org/v1/gonum/mat"
	"log"
	"os"
)

//EMatrix contains data for either MSE or LogLoss loss functions
type EMatrix struct {
	FeaturesInter *mat.Dense
	FeaturesExtra *mat.Dense
	Target        *mat.Dense
	RecordIds     []int
	Description   *string
}

//Sets a description for an EMatrix object
func (ematrix *EMatrix) SetDescription(description string) {
	ematrix.Description = &description
}

//Message prints a message about the current state of the prediction on the current dataset.
//When useLogloss is true, learning curves are reported in logloss space; otherwise RMSE is used.
func (ematrix EMatrix) Message(tree OneTree, testIndex int, testBiases []*mat.Dense, useLogloss bool) float64 {
	currentPrediction := tree.PredictValue(ematrix.FeaturesInter, ematrix.FeaturesExtra)
	if testBiases[testIndex] == nil {
		testBiases[testIndex] = mat.DenseCopyOf(currentPrediction)
	} else {
		testBiases[testIndex].Add(testBiases[testIndex], currentPrediction)
	}

	description := ""
	if ematrix.Description != nil {
		description = *(ematrix.Description)
	}

	var learningCurveValue float64
	if useLogloss {
		// testBiases accumulates raw logits F(x); applySigmoid converts to probabilities for logloss
		learningCurveValue = Logloss(ematrix.Target, testBiases[testIndex], true)
		log.Print("Logloss for ", description, " = ", learningCurveValue)
	} else {
		learningCurveValue = Rmse(ematrix.Target, testBiases[testIndex])
		log.Print("RMSE for ", description, " = ", learningCurveValue)
	}

	return learningCurveValue
}

//ReadEMatrix reads three components of a data set and unites them into one EMatrix object
func ReadEMatrix(fileNameInter, fileNameExtra, fileNameTarget string) (em EMatrix) {
	log.Print("\ttry to load inter <", string(fileNameInter), ">")
	em.FeaturesInter = ReadNpy(fileNameInter)
	log.Print("\ttry to load extra <", string(fileNameExtra), ">")
	em.FeaturesExtra = ReadNpy(fileNameExtra)
	log.Print("\ttry to load Target <", string(fileNameExtra), ">")
	em.Target = ReadNpy(fileNameTarget)

	h := Height(em.FeaturesInter)
	em.RecordIds = make([]int, h)
	for p := 0; p < h; p++ {
		em.RecordIds[p] = p
	}

	return
}

//ReadNpy reads the content of npy file
func ReadNpy(fileName string) (denseMat *mat.Dense) {
	f, err := os.Open(fileName)
	if err != nil {
		log.Fatal(err)
	}
	defer func() { HandleError(f.Close()) }()

	r, err := npyio.NewReader(f)
	if err != nil {
		log.Fatal(err)
	}

	denseMat = &mat.Dense{}
	HandleError(r.Read(denseMat))
	return
}

//Split splits data of receiver by the BestSplit criterion
func (em EMatrix) Split(bias *mat.Dense, split BestSplit) (leftEmatrix, rightEmatrix EMatrix, leftBias, rightBias *mat.Dense) {
	h, w := em.FeaturesInter.Dims()
	_, extraW := em.FeaturesExtra.Dims()
	leftCount, rightCount := 0, 0

	for p := 0; p < h; p++ {
		if em.FeaturesInter.At(p, split.featureIndex) < split.threshold {
			leftCount++
		} else {
			rightCount++
		}
	}

	leftBias = mat.NewDense(leftCount, 1, nil)
	rightBias = mat.NewDense(rightCount, 1, nil)

	leftFeaturesInter := mat.NewDense(leftCount, w, nil)
	rightFeaturesInter := mat.NewDense(rightCount, w, nil)

	leftFeaturesExtra := mat.NewDense(leftCount, extraW, nil)
	rightFeaturesExtra := mat.NewDense(rightCount, extraW, nil)

	leftTarget := mat.NewDense(leftCount, 1, nil)
	rightTarget := mat.NewDense(rightCount, 1, nil)

	leftIds, rightIds := make([]int, 0), make([]int, 0)

	leftInd, rightInd := 0, 0

	for p := 0; p < h; p++ {
		if em.FeaturesInter.At(p, split.featureIndex) < split.threshold {
			leftBias.Set(leftInd, 0, bias.At(p, 0))
			for q := 0; q < w; q++ {
				leftFeaturesInter.Set(leftInd, q, em.FeaturesInter.At(p, q))
			}
			for q := 0; q < extraW; q++ {
				leftFeaturesExtra.Set(leftInd, q, em.FeaturesExtra.At(p, q))
			}
			leftTarget.Set(leftInd, 0, em.Target.At(p, 0))
			leftIds = append(leftIds, em.RecordIds[p])
			leftInd++
		} else {
			rightBias.Set(rightInd, 0, bias.At(p, 0))
			for q := 0; q < w; q++ {
				rightFeaturesInter.Set(rightInd, q, em.FeaturesInter.At(p, q))
			}
			for q := 0; q < extraW; q++ {
				rightFeaturesExtra.Set(rightInd, q, em.FeaturesExtra.At(p, q))
			}
			rightTarget.Set(rightInd, 0, em.Target.At(p, 0))
			rightIds = append(rightIds, em.RecordIds[p])
			rightInd++
		}
	}

	return EMatrix{FeaturesInter: leftFeaturesInter, FeaturesExtra: leftFeaturesExtra, Target: leftTarget, RecordIds: leftIds},
		EMatrix{FeaturesInter: rightFeaturesInter, FeaturesExtra: rightFeaturesExtra, Target: rightTarget, RecordIds: rightIds}, leftBias, rightBias
}

//validateDimensions checks the consistency of dimensions in arrays from the current dataset
//and returns the height (the number of objects), the width (the number of features) and the depth
//(the number of extra features per record) of the current dataset.
func (em EMatrix) validatedDimensions() (h, w, d int) {
	h, w = em.FeaturesInter.Dims()
	extraH, d := em.FeaturesExtra.Dims()
	if extraH != h {
		log.Panicf("the extra height %d is not equal to the inter height %d", extraH, h)
	}
	targetH, targetW := em.Target.Dims()
	if targetH != h {
		log.Panicf("the Target height %d is not equal to the inter height %d", targetH, h)
	}
	if targetW != 1 {
		log.Panicf("the width of Target should be 1 not %d", targetW)
	}
	return h, w, d
}
