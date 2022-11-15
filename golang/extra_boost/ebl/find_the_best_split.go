package extra_boost_lib

import (
	"gonum.org/v1/gonum/mat"
	"gorgonia.org/tensor"
	"log"
	//"log"
)

//BestSplit contains results of the split selection algorithm.
type BestSplit struct {
	bestValue, currentValue          float64
	featureIndex, orderIndex         int
	threshold                        float64
	deltaUp, deltaDown, deltaCurrent *mat.Dense
	validSplit                       bool
	numberOfObjects                  int
}

//OneStepInfo contains information about the algorithm state after passing a cluster of equal values of
//interpolating features
type OneStepInfo struct {
	deltaLoss    float64
	deltaWeight  *mat.Dense
	InterFeature float64
}

//IterateSplits iterates through splits, incrementally updates hessian and gradient and
//calculates optimal weights difference and loss difference.
func IterateSplits(
	indRange IntIterable,
	em *EMatrix,
	q int,
	featuresAs []int,
	bias *mat.Dense,
	currentLoss SplitLoss,
	d int,
	accumGrad *mat.Dense,
	rawHessian *tensor.Dense,
	accumHess *mat.Dense,
	parLambda float64,
	normHess *mat.Dense,
	inverseHess *mat.Dense,
	weight *mat.Dense,
	deltaLoss *mat.Dense,
	unbalancedLoss float64,
) (passInfo []OneStepInfo, totalDeltaLoss float64, totalDeltaWeight *mat.Dense) {
	if !indRange.HasNext() {
		return nil, 0.0, nil
	}

	var currentInd int
	nextInd := indRange.GetNext()
	last := false

	for indRange.HasNext() || !last {
		if indRange.HasNext() {
			currentInd = nextInd
			nextInd = indRange.GetNext()
		} else {
			last = true
		}
		targetVal := em.Target.At(featuresAs[currentInd], 0)
		biasVal := bias.At(featuresAs[currentInd], 0)
		der1 := currentLoss.lossDer1(targetVal, biasVal)
		der2 := currentLoss.lossDer2(targetVal, biasVal)

		for cp := 0; cp < d; cp++ {
			elemGrad := em.FeaturesExtra.At(featuresAs[currentInd], cp)
			accumGrad.Set(cp, 0, accumGrad.At(cp, 0)+der1*elemGrad)

			for cq := 0; cq < d; cq++ {
				element, err := rawHessian.At(featuresAs[currentInd], cp, cq)
				HandleError(err)
				accumHess.Set(cp, cq, der2*element.(float64)+accumHess.At(cp, cq))
				diagEye := 0.0
				if cp == cq {
					diagEye = parLambda
				}
				normHess.Set(cp, cq, accumHess.At(cp, cq)+diagEye)
			}
		}

		if em.FeaturesInter.At(featuresAs[currentInd], q) != em.FeaturesInter.At(featuresAs[nextInd], q) || last {
			HandleError(inverseHess.Inverse(normHess))
			weight.Mul(inverseHess, accumGrad)
			deltaLoss.Mul(weight.T(), accumGrad)

			if passInfo == nil {
				passInfo = make([]OneStepInfo, 0)
			}
			weight.Scale(-1.0, weight)
			oneStepInfo := OneStepInfo{
				deltaLoss:    unbalancedLoss*indRange.DistToMiddle(currentInd) - deltaLoss.At(0, 0),
				deltaWeight:  mat.DenseCopyOf(weight),
				InterFeature: em.FeaturesInter.At(featuresAs[currentInd], q),
			}

			if !last {
				passInfo = append(passInfo, oneStepInfo)
			} else {
				totalDeltaLoss = oneStepInfo.deltaLoss
				totalDeltaWeight = mat.DenseCopyOf(oneStepInfo.deltaWeight)
			}
		}
		currentInd = nextInd
	}
	return
}

//flushIntermediate flushes the gradient and the hessian
func flushIntermediate(d int, accumGrad *mat.Dense, accumHess *mat.Dense) {
	for zeroIndP := 0; zeroIndP < d; zeroIndP++ {
		accumGrad.Set(zeroIndP, 0, 0)
		for zeroIndQ := 0; zeroIndQ < d; zeroIndQ++ {
			accumHess.Set(zeroIndP, zeroIndQ, 0)
		}
	}
}

//selectTheBestSplit scans through different splits and selects the best one
func selectTheBestSplit(em EMatrix, featuresAs []int, bestSplit *BestSplit, h, q, d int, deltasLossUp, deltasLossDown, weightsUp, weightsDown *mat.Dense) {
	firstIter := true

	bestSplit.numberOfObjects = Height(em.FeaturesInter)
	bestSplit.featureIndex = q
	for hInd := 0; hInd < h-1; hInd++ {
		for ; hInd < h-1 && em.FeaturesInter.At(featuresAs[hInd], q) == em.FeaturesInter.At(featuresAs[hInd+1], q); hInd++ {
		}
		if hInd == h-1 {
			break
		}
		currentLossValue := deltasLossUp.At(hInd, 0) + deltasLossDown.At(hInd+1, 0)
		if hInd < h-1 && (firstIter || bestSplit.bestValue > currentLossValue) {
			firstIter = false
			bestSplit.bestValue = currentLossValue
			for qInd := 0; qInd < d; qInd++ {
				bestSplit.deltaUp.Set(qInd, 0, weightsUp.At(hInd, qInd))
				bestSplit.deltaDown.Set(qInd, 0, weightsDown.At(hInd+1, qInd))
			}
			bestSplit.threshold = (em.FeaturesInter.At(featuresAs[hInd], q) + em.FeaturesInter.At(featuresAs[hInd+1], q)) / 2
			bestSplit.orderIndex = hInd
		}
	}
	for ind := 0; ind < d; ind++ {
		bestSplit.deltaCurrent.Set(ind, 0, weightsUp.At(h-1, ind))
	}
	bestSplit.currentValue = deltasLossDown.At(h-1, 0)
	bestSplit.validSplit = !firstIter
}

//selectTheBestSplitCluster scans through different splits and selects the best one
func selectTheBestSplitCluster(em EMatrix, bestSplit *BestSplit, q int, DownPassInfo, UpPassInfo []OneStepInfo) {
	firstIter := true

	if len(DownPassInfo) != len(UpPassInfo) {
		log.Panic("Different dimensions of up and down pass infos")
	}
	h := len(DownPassInfo)

	bestSplit.numberOfObjects = Height(em.FeaturesInter)
	bestSplit.featureIndex = q

	for hInd := 0; hInd < h; hInd++ {
		currentLossValue := DownPassInfo[hInd].deltaLoss + UpPassInfo[h-1-hInd].deltaLoss
		if firstIter || bestSplit.bestValue > currentLossValue {
			firstIter = false
			bestSplit.bestValue = currentLossValue
			bestSplit.deltaUp.CloneFrom(DownPassInfo[hInd].deltaWeight)
			bestSplit.deltaDown.CloneFrom(UpPassInfo[h-1-hInd].deltaWeight)
			bestSplit.threshold = (DownPassInfo[hInd].InterFeature + UpPassInfo[h-1-hInd].InterFeature) / 2.0
			bestSplit.orderIndex = hInd
		}
	}
	bestSplit.validSplit = !firstIter
}

//scanForSplit allocates memory, performs argsort of selected feature column,
//iterates through splits upside down and downside up and selects the best split
//in the current column.
//func scanForSplit(
//	em EMatrix,
//	h, d, q int,
//	bias *mat.Dense,
//	lossFunction SplitLoss,
//	parLambda float64,
//	rawHessian *tensor.Dense,
//	unbalancedLoss float64,
//) (bestSplit BestSplit) {
//	featuresAs := columnArgsort(em.FeaturesInter.ColView(q))
//
//	accumHess := mat.NewDense(d, d, nil)
//	normHess := mat.NewDense(d, d, nil)
//	inverseHess := mat.NewDense(d, d, nil)
//
//	accumGrad := mat.NewDense(d, 1, nil)
//	weight := mat.NewDense(d, 1, nil)
//	deltaLoss := mat.NewDense(1, 1, nil)
//
//	deltasLossUp := mat.NewDense(h, 1, nil)
//	weightsUp := mat.NewDense(h, d, nil)
//	deltasLossDown := mat.NewDense(h, 1, nil)
//	weightsDown := mat.NewDense(h, d, nil)
//
//	bestSplit.deltaUp = mat.NewDense(d, 1, nil)
//	bestSplit.deltaDown = mat.NewDense(d, 1, nil)
//	bestSplit.deltaCurrent = mat.NewDense(d, 1, nil)
//
//	IterateSplits(NewRange(0, h, 1), &em, q, featuresAs,
//		bias, lossFunction, d, accumGrad, rawHessian, accumHess, parLambda,
//		normHess, inverseHess, weight, deltaLoss, deltasLossUp, weightsUp, unbalancedLoss)
//
//	flushIntermediate(d, accumGrad, accumHess)
//
//	IterateSplits(NewRange(h-1, -1, -1), &em, q, featuresAs,
//		bias, lossFunction, d, accumGrad, rawHessian, accumHess, parLambda,
//		normHess, inverseHess, weight, deltaLoss, deltasLossDown, weightsDown, unbalancedLoss)
//
//	selectTheBestSplit(em, featuresAs, &bestSplit, h, q, d, deltasLossUp, deltasLossDown, weightsUp, weightsDown)
//
//	return
//}

//scanForSplit allocates memory, performs argsort of selected feature column,
//iterates through splits upside down and downside up and selects the best split
//in the current column.
func scanForSplitCluster(
	em EMatrix,
	h, d, q int,
	bias *mat.Dense,
	lossFunction SplitLoss,
	parLambda float64,
	rawHessian *tensor.Dense,
	unbalancedLoss float64,
) (bestSplit BestSplit) {
	featuresAs := columnArgsort(em.FeaturesInter.ColView(q))

	accumHess := mat.NewDense(d, d, nil)
	normHess := mat.NewDense(d, d, nil)
	inverseHess := mat.NewDense(d, d, nil)

	accumGrad := mat.NewDense(d, 1, nil)
	weight := mat.NewDense(d, 1, nil)
	deltaLoss := mat.NewDense(1, 1, nil)

	bestSplit.deltaUp = mat.NewDense(d, 1, nil)
	bestSplit.deltaDown = mat.NewDense(d, 1, nil)
	bestSplit.deltaCurrent = mat.NewDense(d, 1, nil)

	var DownPassInfo []OneStepInfo

	DownPassInfo, bestSplit.currentValue, bestSplit.deltaCurrent = IterateSplits(NewRange(0, h, 1), &em, q, featuresAs,
		bias, lossFunction, d, accumGrad, rawHessian, accumHess, parLambda,
		normHess, inverseHess, weight, deltaLoss, unbalancedLoss)

	flushIntermediate(d, accumGrad, accumHess)

	UpPassInfo, _, _ := IterateSplits(NewRange(h-1, -1, -1), &em, q, featuresAs,
		bias, lossFunction, d, accumGrad, rawHessian, accumHess, parLambda,
		normHess, inverseHess, weight, deltaLoss, unbalancedLoss)

	selectTheBestSplitCluster(em, &bestSplit, q, DownPassInfo, UpPassInfo)

	return
}

//allocateArrays allocates the raw hessian array.
func (em EMatrix) allocateArrays() (rawHessian *tensor.Dense) {
	h, _ := em.FeaturesInter.Dims()
	_, d := em.FeaturesExtra.Dims()

	rawHessian = tensor.New(tensor.WithShape(h, d, d), tensor.Of(tensor.Float64))
	for p := 0; p < h; p++ {
		for q := 0; q < d; q++ {
			for r := 0; r < d; r++ {
				HandleError(rawHessian.SetAt(em.FeaturesExtra.At(p, q)*em.FeaturesExtra.At(p, r), p, q, r))
			}
		}
	}
	return
}
