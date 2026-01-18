package poissonlegacy

import (
	"errors"
	"sort"
)

func Split(matrix *PMatrix, bias []float64, params TreeBuildParams) (*SplitResult, error) {
	if matrix == nil {
		return nil, errors.New("nil matrix")
	}
	lossInfo, err := matrix.WholeLoss(params, bias)
	if err != nil {
		return nil, err
	}
	rows := len(lossInfo.TotalLoss)
	if rows == 0 {
		return nil, nil
	}
	cols := len(lossInfo.TotalLoss[0])
	bestIndex := make([]int, cols)
	minLoss := make([]float64, cols)
	for j := 0; j < cols; j++ {
		minLoss[j] = lossInfo.TotalLoss[0][j]
		bestIndex[j] = 0
		for i := 1; i < rows; i++ {
			if lossInfo.TotalLoss[i][j] < minLoss[j] {
				minLoss[j] = lossInfo.TotalLoss[i][j]
				bestIndex[j] = i
			}
		}
	}

	features := make([]int, cols)
	for j := 0; j < cols; j++ {
		features[j] = j
	}
	sort.Slice(features, func(i, j int) bool {
		return minLoss[features[i]] < minLoss[features[j]]
	})

	for _, feature := range features {
		idx := bestIndex[feature]
		sortedValues := make([]float64, len(lossInfo.SortedThresholds))
		for i := 0; i < len(sortedValues); i++ {
			sortedValues[i] = lossInfo.SortedThresholds[i][feature]
		}
		thr, ok := smartThreshold(sortedValues, idx)
		if !ok {
			continue
		}
		leftMask := make([]bool, len(matrix.FeaturesInter))
		rightMask := make([]bool, len(matrix.FeaturesInter))
		for i := 0; i < len(matrix.FeaturesInter); i++ {
			if matrix.FeaturesInter[i][feature] < thr {
				leftMask[i] = true
			} else {
				rightMask[i] = true
			}
		}

		deltaUp := append([]float64(nil), lossInfo.DeltaUp[idx][feature]...)
		deltaDown := append([]float64(nil), lossInfo.DeltaDown[idx][feature]...)
		var deltaCurrent []float64
		if len(lossInfo.DeltaCurrent) > feature {
			deltaCurrent = append([]float64(nil), lossInfo.DeltaCurrent[feature]...)
		}

		return &SplitResult{
			LeftMask:      leftMask,
			RightMask:     rightMask,
			DeltaUp:       deltaUp,
			DeltaDown:     deltaDown,
			Threshold:     thr,
			FeatureIndex:  feature,
			FeatureOffset: idx,
			TotalHeight:   rows,
			DeltaCurrent:  deltaCurrent,
		}, nil
	}
	return nil, nil
}
