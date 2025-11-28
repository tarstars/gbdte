package ebl

import (
	"encoding/json"
	"fmt"
	"github.com/goccy/go-graphviz"
	"gonum.org/v1/gonum/mat"
	"log"
	"os"
	"path"
)

//PredictOperator infers operator that converts extra features into a prediction.
func (oneTree OneTree) PredictOperator(featuresInter *mat.Dense) (prediction *mat.Dense) {
	h, _ := featuresInter.Dims()
	prediction = mat.NewDense(h, oneTree.D, nil)

	for p := 0; p < h; p++ {
		ind := 0
		for oneTree.TreeNodes[ind].LeafIndex == -1 {
			if featuresInter.At(p, oneTree.TreeNodes[ind].FeatureNumber) < oneTree.TreeNodes[ind].Threshold {
				ind = oneTree.TreeNodes[ind].LeftIndex
			} else {
				ind = oneTree.TreeNodes[ind].RightIndex
			}
		}
		prediction.SetRow(p, oneTree.LeafNodes[oneTree.TreeNodes[ind].LeafIndex].Prediction)
	}

	return
}

// PredictValue infers values of a model by inferring an operator and applying it to the Extra data.
func (oneTree OneTree) PredictValue(featuresInter, featuresExtra *mat.Dense) (prediction *mat.Dense) {
	operator := oneTree.PredictOperator(featuresInter)
	h, _ := featuresInter.Dims()
	prediction = mat.NewDense(h, 1, nil)
	for p := 0; p < h; p++ {
		s := 0.0
		for q := 0; q < oneTree.D; q++ {
			s += operator.At(p, q) * featuresExtra.At(p, q)
		}
		prediction.Set(p, 0, s)
	}
	return
}

//EBooster is the model class.
type EBooster struct {
	Trees               []OneTree
	LearningCurveTitles []string
}

//EBoosterParams collect arguments required to construct a booster.
type EBoosterParams struct {
	Matrix         EMatrix
	NStages        int
	RegLambda      float64
	MaxDepth       int
	LearningRate   float64
	LossKind       SplitLoss
	PrintMessages  []EMatrix
	ThreadsNum     int
	UnbalancedLoss float64
	Bias           *mat.Dense
}

//NewEBooster creates a new model.
func NewEBooster(params EBoosterParams) (ebooster *EBooster) {
	ebooster = &EBooster{make([]OneTree, 0), make([]string, 0)}
	h, _ := params.Matrix.FeaturesInter.Dims()
	bias := params.Bias
	if bias == nil {
		bias = mat.NewDense(h, 1, nil)
	}

	var testBiases []*mat.Dense

	for _, currentMessage := range params.PrintMessages {
		description := ""
		if currentMessage.Description != nil {
			description = *currentMessage.Description
		}
		ebooster.LearningCurveTitles = append(ebooster.LearningCurveTitles, description)
		testBiases = append(testBiases, nil)
	}

	useLogloss := false
	if _, ok := params.LossKind.(LogLoss); ok {
		useLogloss = true
	}

	for stage := 0; stage < params.NStages; stage++ {
		log.Printf("Tree number %d\n", stage+1)
		tree := NewTree(params.Matrix, bias, params.RegLambda, params.MaxDepth, params.LearningRate, params.LossKind, params.ThreadsNum, params.UnbalancedLoss)
		deltaB := tree.PredictValue(params.Matrix.FeaturesInter, params.Matrix.FeaturesExtra)
		bias.Add(bias, deltaB)
		currentTreeIndex := len(ebooster.Trees)
		ebooster.Trees = append(ebooster.Trees, tree)
		for testIndex, currentEmatrix := range params.PrintMessages {
			learningCurveValue := currentEmatrix.Message(tree, testIndex, testBiases, useLogloss)
			ebooster.Trees[currentTreeIndex].LearningCurveRow = append(ebooster.Trees[currentTreeIndex].LearningCurveRow, learningCurveValue)
		}
	}
	return
}

//PredictValue infers values of the Target. It requires both sets of features - interpolating and extrapolating.
func (ebooster EBooster) PredictValue(featuresInter, featuresExtra *mat.Dense, treesNumber *int) (prediction *mat.Dense) {
	prediction = ebooster.Trees[0].PredictValue(featuresInter, featuresExtra)

	var n int
	if treesNumber == nil {
		n = len(ebooster.Trees)
	} else {
		n = *treesNumber
	}

	for treeInd := 1; treeInd < n; treeInd++ {
		deltaPrediction := ebooster.Trees[treeInd].PredictValue(featuresInter, featuresExtra)
		prediction.Add(prediction, deltaPrediction)
	}

	return
}

func (ebooster EBooster) Save(filename string) {
	dest, err := os.Create(filename)
	if err != nil {
		log.Print("can't open file ", filename, " to write")
	}
	HandleError(err)
	defer func() { HandleError(dest.Close()) }()

	modelByteRepr, err := json.MarshalIndent(ebooster, "", "  ")
	HandleError(err)

	_, err = dest.Write(modelByteRepr)
	HandleError(err)
}

func (ebooster EBooster) RenderTrees(dumpPrefix, figureType, picturesDirectory string) {
	graphvizType := map[string]graphviz.Format{
		"png": graphviz.PNG,
		"svg": graphviz.SVG,
		"jpg": graphviz.JPG,
	}[figureType]

	for graphInd, currentTree := range ebooster.Trees {
		filename := fmt.Sprintf("%s_%05d.%s", dumpPrefix, graphInd, figureType)
		graphViz, graph := currentTree.DrawGraph()
		HandleError(graphViz.RenderFilename(graph, graphvizType, path.Join(picturesDirectory, filename)))
	}
}

func LoadModel(filename string) (ebooster EBooster) {
	source, err := os.Open(filename)
	HandleError(err)
	defer func() { HandleError(source.Close()) }()

	decoder := json.NewDecoder(source)
	HandleError(decoder.Decode(&ebooster))
	return
}

type LearningCurvesDump struct {
	Titles []string
	Values [][]float64
}

func (ebooster EBooster) DumpLearningCurves(filenameLearningCurves string) {
	destination, err := os.Create(filenameLearningCurves)
	HandleError(err)

	var learningCurvesDump LearningCurvesDump

	learningCurvesDump.Titles = ebooster.LearningCurveTitles
	learningCurvesDump.Values = make([][]float64, 0)

	for _, currentTree := range ebooster.Trees {
		learningCurvesDump.Values = append(learningCurvesDump.Values, currentTree.LearningCurveRow)
	}

	bytesResult, err := json.MarshalIndent(learningCurvesDump, "", "  ")
	HandleError(err)
	_, err = destination.Write(bytesResult)
	HandleError(err)
}
