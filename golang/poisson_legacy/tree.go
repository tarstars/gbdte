package poissonlegacy

import "errors"

type TreeNode struct {
	Left         *TreeNode
	Right        *TreeNode
	FeatureIndex int
	Threshold    float64
	LeafValue    []float64
	Depth        int
}

func (t *TreeNode) IsLeaf() bool {
	return t != nil && t.Left == nil && t.Right == nil
}

func (t *TreeNode) Predict(featuresInter []float64, featuresExtra []float64) (float64, error) {
	if t == nil {
		return 0, errors.New("nil tree")
	}
	if t.IsLeaf() {
		if len(t.LeafValue) == 0 {
			return 0, errors.New("empty leaf value")
		}
		if len(t.LeafValue) == 1 {
			return t.LeafValue[0], nil
		}
		if featuresExtra == nil {
			return 0, errors.New("missing extra features for vector leaf")
		}
		if len(featuresExtra) != len(t.LeafValue) {
			return 0, errors.New("extra feature dimension mismatch")
		}
		value := 0.0
		for i, w := range t.LeafValue {
			value += w * featuresExtra[i]
		}
		return value, nil
	}
	if t.FeatureIndex < 0 || t.FeatureIndex >= len(featuresInter) {
		return 0, errors.New("feature index out of range")
	}
	if featuresInter[t.FeatureIndex] < t.Threshold {
		return t.Left.Predict(featuresInter, featuresExtra)
	}
	return t.Right.Predict(featuresInter, featuresExtra)
}

func buildTree(params TreeBuildParams, matrix *PMatrix, bias []float64) (*TreeNode, error) {
	return buildTreeHelper(params, matrix, bias, nil, nil)
}

func buildTreeHelper(params TreeBuildParams, matrix *PMatrix, bias []float64, parent *TreeNode, prediction []float64) (*TreeNode, error) {
	if priorFinish(params, parent) {
		return leafFromPrediction(prediction, params, bias == nil)
	}

	node := &TreeNode{Depth: 1}
	if parent != nil {
		node.Depth = parent.Depth + 1
	}

	splitResult, err := Split(matrix, bias, params)
	if err != nil {
		return nil, err
	}
	if splitResult == nil {
		return leafFromPrediction(prediction, params, bias == nil)
	}

	leftMatrix, err := matrix.GetSlice(splitResult.LeftMask)
	if err != nil {
		return nil, err
	}
	rightMatrix, err := matrix.GetSlice(splitResult.RightMask)
	if err != nil {
		return nil, err
	}

	if leftMatrix.Height() < 2 || rightMatrix.Height() < 2 {
		return leafFromPrediction(prediction, params, bias == nil)
	}

	node.FeatureIndex = splitResult.FeatureIndex
	node.Threshold = splitResult.Threshold

	var leftBias []float64
	var rightBias []float64
	if bias != nil {
		leftBias = sliceFloat(bias, splitResult.LeftMask)
		rightBias = sliceFloat(bias, splitResult.RightMask)
	}

	leftNode, err := buildTreeHelper(params, leftMatrix, leftBias, node, splitResult.DeltaUp)
	if err != nil {
		return nil, err
	}
	rightNode, err := buildTreeHelper(params, rightMatrix, rightBias, node, splitResult.DeltaDown)
	if err != nil {
		return nil, err
	}

	node.Left = leftNode
	node.Right = rightNode
	return node, nil
}

func priorFinish(params TreeBuildParams, parent *TreeNode) bool {
	if parent == nil {
		return false
	}
	return params.MaxDepth <= parent.Depth
}

func leafFromPrediction(prediction []float64, params TreeBuildParams, firstTree bool) (*TreeNode, error) {
	if prediction == nil {
		return nil, errors.New("missing leaf prediction")
	}
	scale := params.LearningRate
	if firstTree {
		scale = 1.0
	}
	leaf := &TreeNode{}
	leaf.LeafValue = make([]float64, len(prediction))
	for i, v := range prediction {
		leaf.LeafValue[i] = v * scale
	}
	return leaf, nil
}
