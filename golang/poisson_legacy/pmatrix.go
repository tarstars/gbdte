package poissonlegacy

import (
	"errors"
	"math"
)

type PMatrix struct {
	Bjids         []int
	Freqs         []float64
	FeaturesInter [][]float64
	FeaturesExtra [][]float64
	Time          []float64
	Functions     []TimeFunc
	Psi           []float64
	Gax           [][]int
}

func NewPMatrixFromDense(
	bjids []int,
	freqs []float64,
	featuresInter [][]float64,
	featuresExtra [][]float64,
	psi []float64,
) (*PMatrix, error) {
	if len(bjids) != len(freqs) || len(bjids) != len(featuresInter) {
		return nil, errors.New("bjids, freqs, and features_inter must have the same length")
	}
	if len(featuresInter) == 0 {
		return nil, errors.New("empty features_inter")
	}
	cols := len(featuresInter[0])
	for _, row := range featuresInter {
		if len(row) != cols {
			return nil, errors.New("inconsistent features_inter width")
		}
	}

	freqCopy := make([]float64, len(freqs))
	for i, v := range freqs {
		if v < 1 {
			v = 1
		}
		freqCopy[i] = v
	}

	interCopy := make([][]float64, len(featuresInter))
	for i, row := range featuresInter {
		cloned := make([]float64, len(row))
		copy(cloned, row)
		interCopy[i] = cloned
	}

	var extraCopy [][]float64
	if featuresExtra != nil {
		if len(featuresExtra) != len(featuresInter) {
			return nil, errors.New("features_extra length mismatch")
		}
		depth := len(featuresExtra[0])
		for _, row := range featuresExtra {
			if len(row) != depth {
				return nil, errors.New("inconsistent features_extra width")
			}
		}
		if psi != nil && len(psi) != depth {
			return nil, errors.New("psi length mismatch")
		}
		extraCopy = make([][]float64, len(featuresExtra))
		for i, row := range featuresExtra {
			cloned := make([]float64, len(row))
			copy(cloned, row)
			extraCopy[i] = cloned
		}
	}

	matrix := &PMatrix{
		Bjids:         append([]int(nil), bjids...),
		Freqs:         freqCopy,
		FeaturesInter: interCopy,
		FeaturesExtra: extraCopy,
		Psi:           append([]float64(nil), psi...),
	}

	gax, err := makeGax(matrix.FeaturesInter)
	if err != nil {
		return nil, err
	}
	matrix.Gax = gax
	return matrix, nil
}

func NewPMatrixFromIterables(
	bjids []int,
	freqs []float64,
	featuresInter [][]float64,
	functions []TimeFunc,
	tRange []float64,
	time []float64,
) (*PMatrix, error) {
	if len(bjids) != len(freqs) || len(bjids) != len(featuresInter) {
		return nil, errors.New("bjids, freqs, and features_inter must have the same length")
	}
	freqCopy := make([]float64, len(freqs))
	for i, v := range freqs {
		if v < 1 {
			v = 1
		}
		freqCopy[i] = v
	}
	featuresCopy := make([][]float64, len(featuresInter))
	for i, row := range featuresInter {
		cloned := make([]float64, len(row))
		copy(cloned, row)
		featuresCopy[i] = cloned
	}

	matrix := &PMatrix{
		Bjids:         append([]int(nil), bjids...),
		Freqs:         freqCopy,
		FeaturesInter: featuresCopy,
		Functions:     functions,
	}

	if time != nil {
		if len(time) != len(bjids) {
			return nil, errors.New("time length mismatch")
		}
		matrix.Time = append([]float64(nil), time...)
	}

	if len(functions) > 0 && len(time) > 0 {
		featuresExtra := make([][]float64, len(time))
		for i, t := range time {
			row := make([]float64, len(functions))
			for j, fn := range functions {
				row[j] = fn(t)
			}
			featuresExtra[i] = row
		}
		matrix.FeaturesExtra = featuresExtra
		if len(tRange) >= 2 {
			psi := make([]float64, len(functions))
			dt := tRange[1] - tRange[0]
			for _, t := range tRange {
				for j, fn := range functions {
					psi[j] += fn(t)
				}
			}
			for j := range psi {
				psi[j] *= dt
			}
			matrix.Psi = psi
		}
	}

	gax, err := makeGax(matrix.FeaturesInter)
	if err != nil {
		return nil, err
	}
	matrix.Gax = gax
	return matrix, nil
}

func (p *PMatrix) Height() int {
	return len(p.Bjids)
}

func (p *PMatrix) GetSlice(mask []bool) (*PMatrix, error) {
	if len(mask) != len(p.Bjids) {
		return nil, errors.New("mask length mismatch")
	}
	featuresInter := slice2D(p.FeaturesInter, mask)
	featuresExtra := sliceOptional2D(p.FeaturesExtra, mask)
	matrix := &PMatrix{
		Bjids:         sliceInt(p.Bjids, mask),
		Freqs:         sliceFloat(p.Freqs, mask),
		FeaturesInter: featuresInter,
		FeaturesExtra: featuresExtra,
		Time:          sliceOptionalFloat(p.Time, mask),
		Functions:     p.Functions,
		Psi:           p.Psi,
	}
	gax, err := makeGax(matrix.FeaturesInter)
	if err != nil {
		return nil, err
	}
	matrix.Gax = gax
	return matrix, nil
}

func (p *PMatrix) WholeLoss(params TreeBuildParams, bias []float64) (LossResult, error) {
	if bias == nil {
		if p.FeaturesExtra == nil {
			return p.wholeLossFirstTree(params)
		}
		return p.wholeLossFirstTreeExtra(params)
	}
	if p.FeaturesExtra == nil {
		return p.wholeLossNextTree(params, bias)
	}
	return p.wholeLossNextTreeExtra(params, bias)
}

func (p *PMatrix) wholeLossFirstTree(params TreeBuildParams) (LossResult, error) {
	rows := len(p.Freqs)
	if rows == 0 {
		return LossResult{}, errors.New("empty dataset")
	}
	cols := len(p.FeaturesInter[0])
	freqsRect := gatherVectorByGax(p.Freqs, p.Gax)
	freqsCum := cumsumForward2D(freqsRect)
	lambdaForward := make2D(rows, cols)
	lossForward := make2D(rows, cols)
	for i := 0; i < rows; i++ {
		count := float64(i + 1)
		for j := 0; j < cols; j++ {
			lambda := freqsCum[i][j] / count
			lambdaForward[i][j] = lambda
			lossForward[i][j] = count*lambda - math.Log(lambda)*freqsCum[i][j]
		}
	}

	freqsCumBackward := make2D(rows-1, cols)
	objectsBackward := make2D(rows-1, cols)
	for i := 0; i < rows-1; i++ {
		count := float64(rows - 1 - i)
		for j := 0; j < cols; j++ {
			freqsCumBackward[i][j] = freqsCum[rows-1][j] - freqsCum[i][j]
			objectsBackward[i][j] = count
		}
	}

	lambdaBackward := make2D(rows-1, cols)
	lossBackward := make2D(rows-1, cols)
	for i := 0; i < rows-1; i++ {
		for j := 0; j < cols; j++ {
			lambda := freqsCumBackward[i][j] / objectsBackward[i][j]
			lambdaBackward[i][j] = lambda
			lossBackward[i][j] = objectsBackward[i][j]*lambda - math.Log(lambda)*freqsCumBackward[i][j]
		}
	}

	splitLoss := make2D(rows-1, cols)
	for i := 0; i < rows-1; i++ {
		for j := 0; j < cols; j++ {
			splitLoss[i][j] = lossForward[i][j] + lossBackward[i][j]
		}
	}

	penalty := edgePenalty(rows)
	for i := 0; i < rows-1; i++ {
		for j := 0; j < cols; j++ {
			splitLoss[i][j] += penalty[i] * params.UnbalancedPenalty
		}
	}

	sortedThresholds := takeAlongAxis(p.FeaturesInter, p.Gax)
	upper := make3D(rows-1, cols, 1)
	lower := make3D(rows-1, cols, 1)
	for i := 0; i < rows-1; i++ {
		for j := 0; j < cols; j++ {
			upper[i][j][0] = lambdaForward[i][j]
			lower[i][j][0] = lambdaBackward[i][j]
		}
	}
	deltaCurrent := make2D(cols, 1)
	for j := 0; j < cols; j++ {
		deltaCurrent[j][0] = lambdaForward[rows-1][j]
	}

	return LossResult{
		TotalLoss:        splitLoss,
		SortedThresholds: sortedThresholds,
		DeltaUp:          upper,
		DeltaDown:        lower,
		DeltaCurrent:     deltaCurrent,
	}, nil
}

func (p *PMatrix) wholeLossFirstTreeExtra(params TreeBuildParams) (LossResult, error) {
	if p.FeaturesExtra == nil {
		return LossResult{}, errors.New("missing extra features")
	}
	rows := len(p.Freqs)
	cols := len(p.FeaturesInter[0])
	depth := len(p.FeaturesExtra[0])
	featuresExtraRect := gatherExtraByGax(p.FeaturesExtra, p.Gax)
	sortedThresholds := takeAlongAxis(p.FeaturesInter, p.Gax)

	leftProduct := make4D(rows, cols, depth)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			for a := 0; a < depth; a++ {
				for b := 0; b < depth; b++ {
					leftProduct[i][j][a][b] = featuresExtraRect[i][j][a] * featuresExtraRect[i][j][b]
				}
			}
		}
	}

	targetRect := gatherVectorByGax(p.Freqs, p.Gax)
	rightPart := make3D(rows, cols, depth)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			for d := 0; d < depth; d++ {
				rightPart[i][j][d] = targetRect[i][j] * featuresExtraRect[i][j][d]
			}
		}
	}

	leftForward := cumsumForward4D(leftProduct)
	leftBackward := cumsumBackward4D(leftProduct)
	rightForward := cumsumForward3D(rightPart)
	rightBackward := cumsumBackward3D(rightPart)

	weightsForward := make3D(rows, cols, depth)
	weightsBackward := make3D(rows, cols, depth)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			wForward, err := solveLinearSystem(leftForward[i][j], rightForward[i][j], params.RegLambda)
			if err != nil {
				return LossResult{}, err
			}
			wBackward, err := solveLinearSystem(leftBackward[i][j], rightBackward[i][j], params.RegLambda)
			if err != nil {
				return LossResult{}, err
			}
			copy(weightsForward[i][j], wForward)
			copy(weightsBackward[i][j], wBackward)
		}
	}

	lossForward := make2D(rows, cols)
	lossBackward := make2D(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			forward := 0.0
			backward := 0.0
			for d := 0; d < depth; d++ {
				forward -= weightsForward[i][j][d] * rightForward[i][j][d]
				backward -= weightsBackward[i][j][d] * rightBackward[i][j][d]
			}
			lossForward[i][j] = forward
			lossBackward[i][j] = backward
		}
	}

	totalLoss := make2D(rows-1, cols)
	for i := 0; i < rows-1; i++ {
		for j := 0; j < cols; j++ {
			totalLoss[i][j] = lossForward[i][j] + lossBackward[i+1][j]
		}
	}

	deltaCurrent := make2D(cols, depth)
	for j := 0; j < cols; j++ {
		copy(deltaCurrent[j], weightsForward[rows-1][j])
	}

	return LossResult{
		TotalLoss:        totalLoss,
		SortedThresholds: sortedThresholds,
		DeltaUp:          weightsForward,
		DeltaDown:        weightsBackward[1:],
		DeltaCurrent:     deltaCurrent,
	}, nil
}

func (p *PMatrix) wholeLossNextTree(params TreeBuildParams, bias []float64) (LossResult, error) {
	rows := len(p.Freqs)
	if len(bias) != rows {
		return LossResult{}, errors.New("bias length mismatch")
	}
	cols := len(p.FeaturesInter[0])
	frac := make([]float64, rows)
	fracSq := make([]float64, rows)
	for i := 0; i < rows; i++ {
		frac[i] = p.Freqs[i] / bias[i]
		fracSq[i] = frac[i] / bias[i]
	}

	fracRect := gatherVectorByGax(frac, p.Gax)
	fracSqRect := gatherVectorByGax(fracSq, p.Gax)
	deltaFreqRect := make2D(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			deltaFreqRect[i][j] = 1.0 - fracRect[i][j]
		}
	}

	csForward := cumsumForward2D(deltaFreqRect)
	csBackward := cumsumBackward2D(deltaFreqRect)
	csSqForward := cumsumForward2D(fracSqRect)
	csSqBackward := cumsumBackward2D(fracSqRect)

	deltaForward := make2D(rows, cols)
	deltaBackward := make2D(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			denForward := csSqForward[i][j] + params.RegLambda
			denBackward := csSqBackward[i][j] + params.RegLambda
			deltaForward[i][j] = -csForward[i][j] / denForward
			deltaBackward[i][j] = -csBackward[i][j] / denBackward
		}
	}

	biasRect := gatherVectorByGax(bias, p.Gax)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			minDelta := params.RegLambda - biasRect[i][j]
			if deltaForward[i][j] < minDelta {
				deltaForward[i][j] = minDelta
			}
			if deltaBackward[i][j] < minDelta {
				deltaBackward[i][j] = minDelta
			}
		}
	}

	bestValueForward := make2D(rows, cols)
	bestValueBackward := make2D(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			bestValueForward[i][j] = -(csForward[i][j] * csForward[i][j]) / (csSqForward[i][j] + params.RegLambda)
			bestValueBackward[i][j] = -(csBackward[i][j] * csBackward[i][j]) / (csSqBackward[i][j] + params.RegLambda)
		}
	}

	splitLoss := make2D(rows-1, cols)
	upper := make3D(rows-1, cols, 1)
	lower := make3D(rows-1, cols, 1)
	for i := 0; i < rows-1; i++ {
		for j := 0; j < cols; j++ {
			splitLoss[i][j] = bestValueForward[i][j] + bestValueBackward[i+1][j]
			upper[i][j][0] = deltaForward[i][j]
			lower[i][j][0] = deltaBackward[i+1][j]
		}
	}

	sortedThresholds := takeAlongAxis(p.FeaturesInter, p.Gax)
	deltaCurrent := make2D(cols, 1)
	for j := 0; j < cols; j++ {
		deltaCurrent[j][0] = deltaForward[rows-1][j]
	}

	return LossResult{
		TotalLoss:        splitLoss,
		SortedThresholds: sortedThresholds,
		DeltaUp:          upper,
		DeltaDown:        lower,
		DeltaCurrent:     deltaCurrent,
	}, nil
}

func (p *PMatrix) wholeLossNextTreeExtra(params TreeBuildParams, bias []float64) (LossResult, error) {
	if p.FeaturesExtra == nil {
		return LossResult{}, errors.New("missing extra features")
	}
	if p.Psi == nil || len(p.Psi) == 0 {
		return LossResult{}, errors.New("missing psi for extra Poisson loss")
	}
	rows := len(p.Freqs)
	if len(bias) != rows {
		return LossResult{}, errors.New("bias length mismatch")
	}
	cols := len(p.FeaturesInter[0])
	depth := len(p.FeaturesExtra[0])
	featuresExtraRect := gatherExtraByGax(p.FeaturesExtra, p.Gax)
	sortedThresholds := takeAlongAxis(p.FeaturesInter, p.Gax)

	freqRect := gatherVectorByGax(p.Freqs, p.Gax)
	biasRect := gatherVectorByGax(bias, p.Gax)

	objectsForward := countObjects(p.Bjids, p.Gax)
	revGax := reverseGax(p.Gax)
	objectsBackward := countObjects(p.Bjids, revGax)
	objectsBackward = reverse2DFloat(objectsBackward)

	psiForward := make3D(rows, cols, depth)
	psiBackward := make3D(rows, cols, depth)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			for d := 0; d < depth; d++ {
				psiForward[i][j][d] = objectsForward[i][j] * p.Psi[d]
				psiBackward[i][j][d] = objectsBackward[i][j] * p.Psi[d]
			}
		}
	}

	hess := make4D(rows, cols, depth)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			biasSq := biasRect[i][j] * biasRect[i][j]
			for a := 0; a < depth; a++ {
				for b := 0; b < depth; b++ {
					hess[i][j][a][b] = freqRect[i][j] * featuresExtraRect[i][j][a] * featuresExtraRect[i][j][b] / biasSq
				}
			}
		}
	}

	hessForward := cumsumForward4D(hess)
	hessBackward := cumsumBackward4D(hess)

	gradRight := make3D(rows, cols, depth)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			for d := 0; d < depth; d++ {
				gradRight[i][j][d] = freqRect[i][j] * featuresExtraRect[i][j][d] / biasRect[i][j]
			}
		}
	}
	gradForward := cumsumForward3D(gradRight)
	gradBackward := cumsumBackward3D(gradRight)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			for d := 0; d < depth; d++ {
				gradForward[i][j][d] = psiForward[i][j][d] - gradForward[i][j][d]
				gradBackward[i][j][d] = psiBackward[i][j][d] - gradBackward[i][j][d]
			}
		}
	}

	weightsForward := make3D(rows, cols, depth)
	weightsBackward := make3D(rows, cols, depth)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			wForward, err := solveLinearSystem(hessForward[i][j], gradForward[i][j], params.RegLambda)
			if err != nil {
				return LossResult{}, err
			}
			wBackward, err := solveLinearSystem(hessBackward[i][j], gradBackward[i][j], params.RegLambda)
			if err != nil {
				return LossResult{}, err
			}
			for d := 0; d < depth; d++ {
				weightsForward[i][j][d] = -wForward[d]
				weightsBackward[i][j][d] = -wBackward[d]
			}
		}
	}

	lossForward := make2D(rows, cols)
	lossBackward := make2D(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			lf := 0.0
			lb := 0.0
			for d := 0; d < depth; d++ {
				lf += -gradForward[i][j][d] * weightsForward[i][j][d]
				lb += -gradBackward[i][j][d] * weightsBackward[i][j][d]
			}
			lossForward[i][j] = lf
			lossBackward[i][j] = lb
		}
	}

	totalLoss := make2D(rows-1, cols)
	for i := 0; i < rows-1; i++ {
		for j := 0; j < cols; j++ {
			totalLoss[i][j] = lossForward[i][j] + lossBackward[i+1][j]
		}
	}

	deltaCurrent := make2D(cols, depth)
	for j := 0; j < cols; j++ {
		copy(deltaCurrent[j], weightsForward[rows-1][j])
	}

	return LossResult{
		TotalLoss:        totalLoss,
		SortedThresholds: sortedThresholds,
		DeltaUp:          weightsForward[:rows-1],
		DeltaDown:        weightsBackward[1:],
		DeltaCurrent:     deltaCurrent,
	}, nil
}

func reverse2DFloat(data [][]float64) [][]float64 {
	rows := len(data)
	if rows == 0 {
		return nil
	}
	out := make2D(rows, len(data[0]))
	for i := 0; i < rows; i++ {
		copy(out[i], data[rows-1-i])
	}
	return out
}
