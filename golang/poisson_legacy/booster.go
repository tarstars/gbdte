package poissonlegacy

import "errors"

type Booster struct {
	Trees []*TreeNode
}

func Train(params TreeBuildParams, matrix *PMatrix, numBoostRound int) (*Booster, error) {
	if matrix == nil {
		return nil, errors.New("nil matrix")
	}
	booster := &Booster{}
	var bias []float64
	for stage := 0; stage < numBoostRound; stage++ {
		tree, err := buildTree(params, matrix, bias)
		if err != nil {
			return nil, err
		}
		preds, err := predictTree(tree, matrix.FeaturesInter, matrix.FeaturesExtra)
		if err != nil {
			return nil, err
		}
		if bias == nil {
			bias = preds
		} else {
			for i := range bias {
				bias[i] += preds[i]
			}
		}
		if params.CheckZero {
			for _, v := range bias {
				if v == 0 {
					return nil, errors.New("zero prediction")
				}
			}
		}
		booster.Trees = append(booster.Trees, tree)
	}
	return booster, nil
}

func (b *Booster) Predict(featuresInter [][]float64, featuresExtra [][]float64) ([]float64, error) {
	if len(b.Trees) == 0 {
		return nil, errors.New("empty booster")
	}
	rows := len(featuresInter)
	preds := make([]float64, rows)
	for _, tree := range b.Trees {
		for i := 0; i < rows; i++ {
			var extra []float64
			if featuresExtra != nil {
				extra = featuresExtra[i]
			}
			value, err := tree.Predict(featuresInter[i], extra)
			if err != nil {
				return nil, err
			}
			preds[i] += value
		}
	}
	return preds, nil
}

func predictTree(tree *TreeNode, featuresInter [][]float64, featuresExtra [][]float64) ([]float64, error) {
	rows := len(featuresInter)
	preds := make([]float64, rows)
	for i := 0; i < rows; i++ {
		var extra []float64
		if featuresExtra != nil {
			extra = featuresExtra[i]
		}
		val, err := tree.Predict(featuresInter[i], extra)
		if err != nil {
			return nil, err
		}
		preds[i] = val
	}
	return preds, nil
}
