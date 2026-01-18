package poissonlegacy

// TreeBuildParams mirrors the legacy Poisson tree parameters.
type TreeBuildParams struct {
	MaxDepth          int
	LearningRate      float64
	UnbalancedPenalty float64
	RegLambda         float64
	CheckZero         bool
}

type LossResult struct {
	TotalLoss        [][]float64
	SortedThresholds [][]float64
	DeltaUp          [][][]float64
	DeltaDown        [][][]float64
	DeltaCurrent     [][]float64
}

type SplitResult struct {
	LeftMask      []bool
	RightMask     []bool
	DeltaUp       []float64
	DeltaDown     []float64
	Threshold     float64
	FeatureIndex  int
	FeatureOffset int
	TotalHeight   int
	DeltaCurrent  []float64
}
