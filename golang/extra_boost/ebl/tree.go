package ebl

import (
	"fmt"
	"github.com/goccy/go-graphviz"
	"github.com/goccy/go-graphviz/cgraph"
	"gonum.org/v1/gonum/mat"
	"strings"
)

//TreeNode is a node of a tree. Tree is stored in an array. LeftIndex and RightIndex are equal to -1
//when the current node is a leaf otherwise they contain array indices of children.
//A leaf node contains LeafIndex that is an index of the LeafNodes array.
type TreeNode struct {
	TreeNodeId            int
	FeatureNumber         int
	Threshold             float64
	LeftIndex, RightIndex int // -1, -1 if it is a leaf
	LeafIndex             int // -1 if it is a non-leaf tree node
	NumberOfObjects       int
	CurrentLoss           float64
	NoSplit               bool
}

//GraphDescription returns the description of a tree node for tree rendering as a graph
func (node TreeNode) GraphDescription() string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintln("#", node.NumberOfObjects))
	sb.WriteString(fmt.Sprintln("id: ", node.TreeNodeId))
	sb.WriteString(fmt.Sprintln("loss: ", node.CurrentLoss))
	if node.NoSplit {
		sb.WriteString("NoSplit")
	} else {
		sb.WriteString(fmt.Sprintf("f_%d < %6.5f", node.FeatureNumber, node.Threshold))
	}
	return sb.String()
}

func NewTreeNode() TreeNode {
	return TreeNode{0, 0, 0, -1, -1, -1, 0, 0, false}
}

//NewTreeNodeFromSplitInfo creates a new tree node and extract a features index and a split threshold
//from a BestSplit object.
func NewTreeNodeFromSplitInfo(splitInfo BestSplit, treeNodeId int) TreeNode {
	treeNode := NewTreeNode()
	treeNode.TreeNodeId = treeNodeId
	treeNode.FeatureNumber = splitInfo.featureIndex
	treeNode.Threshold = splitInfo.threshold
	treeNode.NumberOfObjects = splitInfo.numberOfObjects
	treeNode.CurrentLoss = splitInfo.currentValue
	return treeNode
}

//IsLeaf returns whether this node is a LeafNode.
func (node TreeNode) IsLeaf() bool {
	return node.LeafIndex != -1
}

//LeafNode stores leaf-related information. It is a prediction from this leaf and possibly some other statistics.
type LeafNode struct {
	LeafNodeId      int
	Prediction      []float64
	RecordIds       []int
	NumberOfObjects int
}

//GraphDescription returns the description of a leaf node for tree rendering as a graph
func (node LeafNode) GraphDescription() string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintln("id: ", node.LeafNodeId))
	sb.WriteString("[")
	for _, val := range node.Prediction {
		sb.WriteString(fmt.Sprintf("  %6.2f,\n", val))
	}
	sb.WriteString("]\n")
	sb.WriteString(fmt.Sprintln(node.NumberOfObjects))
	return sb.String()
}

//NewLeafNode creates a new leaf node.
func NewLeafNode(leafData *mat.Dense, numberOfObjects int, learningRate float64, recordIds []int) (leafNode *LeafNode) {
	h, _ := leafData.Dims()

	leafNode = &LeafNode{LeafNodeId: -1, Prediction: make([]float64, h), NumberOfObjects: numberOfObjects, RecordIds: nil} // RecordIds: RecordIds

	for ind := 0; ind < h; ind++ {
		leafNode.Prediction[ind] = leafData.At(ind, 0) * learningRate
	}

	return
}

//OneTree describes one tree in a classifier.
type OneTree struct {
	D                int // the extra depth
	TreeNodes        []TreeNode
	LeafNodes        []LeafNode
	LearningCurveRow []float64
}

//GetLeafDescription returns the description of a leaf node
func (tree OneTree) GetLeafDescription(ind int) string {
	return tree.LeafNodes[tree.TreeNodes[ind].LeafIndex].GraphDescription()
}

//GetLeafDescription returns the description of a leaf node
func (tree OneTree) GetNodeDescription(ind int) string {
	return tree.TreeNodes[ind].GraphDescription()
}

//NewTree builds one new tree in a model.
func NewTree(ematrix EMatrix, bias *mat.Dense, regLambda float64, maxDepth int, learningRate float64, lossKind SplitLoss, threadsNum int, unbalancedLoss float64) (oneTree OneTree) {
	oneTree.TreeNodes = make([]TreeNode, 0)
	oneTree.LeafNodes = make([]LeafNode, 0)
	_, oneTree.D = ematrix.FeaturesExtra.Dims()

	(&oneTree).BuildTree(ematrix, bias, nil, regLambda, maxDepth, 0, learningRate, lossKind, threadsNum, unbalancedLoss)

	return
}

//BuildTree recurrently builds a tree node.
func (oneTree *OneTree) BuildTree(
	ematrix EMatrix, bias *mat.Dense,
	leafInfo *LeafNode, parLambda float64, maxDepth int, currentDepth int,
	learningRate float64,
	lossKind SplitLoss,
	threadsNum int,
	unbalancedLoss float64,
) int {
	shouldSplit := leafInfo == nil || (currentDepth < maxDepth && Height(ematrix.FeaturesInter) > 5)
	var bestSplit *BestSplit
	if shouldSplit {
		bestSplit = TheBestSplit(ematrix, bias, parLambda, lossKind, threadsNum, unbalancedLoss)
		if bestSplit != nil && bestSplit.validSplit {
			treeNodeId := len(oneTree.TreeNodes)
			currentTreeNode := NewTreeNodeFromSplitInfo(*bestSplit, treeNodeId)
			oneTree.TreeNodes = append(oneTree.TreeNodes, currentTreeNode)

			leftEmatrix, rightEmatrix, leftBias, rightBias := ematrix.Split(bias, *bestSplit)

			leftLeaf := NewLeafNode(bestSplit.deltaUp, Height(leftEmatrix.FeaturesInter), learningRate, leftEmatrix.RecordIds)
			leftNodeId := oneTree.BuildTree(leftEmatrix, leftBias, leftLeaf, parLambda, maxDepth, currentDepth+1, learningRate, lossKind, threadsNum, unbalancedLoss)
			oneTree.TreeNodes[treeNodeId].LeftIndex = leftNodeId

			rightLeaf := NewLeafNode(bestSplit.deltaDown, Height(rightEmatrix.FeaturesInter), learningRate, rightEmatrix.RecordIds)
			rightNodeId := oneTree.BuildTree(rightEmatrix, rightBias, rightLeaf, parLambda, maxDepth, currentDepth+1, learningRate, lossKind, threadsNum, unbalancedLoss)
			oneTree.TreeNodes[treeNodeId].RightIndex = rightNodeId

			return treeNodeId
		}
	}

	markNoSplit := shouldSplit && (bestSplit == nil || !bestSplit.validSplit)
	return oneTree.makeLeafNode(ematrix, leafInfo, learningRate, bestSplit, markNoSplit, parLambda, lossKind)
}

func (oneTree *OneTree) makeLeafNode(
	ematrix EMatrix,
	leafInfo *LeafNode,
	learningRate float64,
	bestSplit *BestSplit,
	markNoSplit bool,
	parLambda float64,
	lossKind SplitLoss,
) int {
	treeNodeId := len(oneTree.TreeNodes)
	currentTreeNode := NewTreeNode()
	currentTreeNode.TreeNodeId = treeNodeId
	currentTreeNode.NumberOfObjects = Height(ematrix.FeaturesInter)
	if bestSplit != nil {
		currentTreeNode.CurrentLoss = bestSplit.currentValue
	}
	if markNoSplit {
		currentTreeNode.NoSplit = true
		currentTreeNode.FeatureNumber = -1
	}
	oneTree.TreeNodes = append(oneTree.TreeNodes, currentTreeNode)

	var leaf *LeafNode
	if leafInfo != nil {
		leaf = leafInfo
	} else {
		var delta *mat.Dense
		if bestSplit != nil && bestSplit.deltaCurrent != nil {
			delta = mat.DenseCopyOf(bestSplit.deltaCurrent)
		} else {
			delta = solveNoSplitDelta(ematrix, parLambda, lossKind, oneTree.D)
		}
		leaf = NewLeafNode(delta, Height(ematrix.FeaturesInter), learningRate, ematrix.RecordIds)
	}
	if leaf.RecordIds == nil && len(ematrix.RecordIds) > 0 {
		leaf.RecordIds = append([]int(nil), ematrix.RecordIds...)
	}
	leafNodeId := len(oneTree.LeafNodes)
	oneTree.TreeNodes[treeNodeId].LeafIndex = leafNodeId
	leaf.LeafNodeId = leafNodeId
	oneTree.LeafNodes = append(oneTree.LeafNodes, *leaf)
	return treeNodeId
}

//TheBestSplit finds the best possible split in the given ematrix.
//This function performs multithreading iteration over columns of the ematrix.
func TheBestSplit(ematrix EMatrix, bias *mat.Dense, parLambda float64, lossKind SplitLoss, threadsNum int, unbalancedLoss float64) *BestSplit {
	h, w, d := ematrix.validatedDimensions()
	rawHessian := ematrix.allocateArrays()

	// log.Printf("ematrix %d\n", h)
	result := make([]BestSplit, w)

	if threadsNum == 1 {
		for q := 0; q < w; q++ {
			result[q] = scanForSplitCluster(ematrix, h, d, q, bias, lossKind, parLambda, rawHessian, unbalancedLoss)
		}
	} else {
		taskPool := NewPool(threadsNum)

		for q := 0; q < w; q++ {
			bestSplitFunc := func(localQ int) BestSplit {
				return scanForSplitCluster(ematrix, h, d, localQ, bias, lossKind, parLambda, rawHessian, unbalancedLoss)
			}
			taskPool.AddTask(&TaskFindBestSplit{result, q, bestSplitFunc})
			//result[q] = scanForSplit(ematrix, h, d, q, bias, lossKind, parLambda, rawHessian)
		}
		taskPool.Close()
		taskPool.WaitAll()
	}

	minimalLoss := 0.0
	bestIndex := 0

	firstTime := true

	//	fmt.Print("losses: ")

	for ind, currentSplit := range result {
		//fmt.Printf("%0.2g ", currentSplit.bestValue)
		if currentSplit.validSplit && (firstTime || minimalLoss > currentSplit.bestValue) {
			firstTime = false
			minimalLoss = currentSplit.bestValue
			bestIndex = ind
		}
	}

	//	fmt.Println()

	if firstTime {
		return nil
	}

	return &result[bestIndex]
}

func recurrentDraw(g *cgraph.Graph, tree OneTree, nodeNumber int, parentNode *cgraph.Node) {
	currentNode, err := g.CreateNode(fmt.Sprint(tree.TreeNodes[nodeNumber].TreeNodeId))
	HandleError(err)

	if parentNode != nil {
		g.CreateEdge("", parentNode, currentNode)
	}

	if tree.TreeNodes[nodeNumber].IsLeaf() {
		currentNode.Set("label", tree.GetLeafDescription(nodeNumber))
		currentNode.Set("shape", "box")
	} else {
		currentNode.Set("label", tree.GetNodeDescription(nodeNumber))
		recurrentDraw(g, tree, tree.TreeNodes[nodeNumber].LeftIndex, currentNode)
		recurrentDraw(g, tree, tree.TreeNodes[nodeNumber].RightIndex, currentNode)
	}
}

func (tree OneTree) DrawGraph() (*graphviz.Graphviz, *cgraph.Graph) {
	graphViz := graphviz.New()
	graph, err := graphViz.Graph()
	HandleError(err)

	recurrentDraw(graph, tree, 0, nil)

	return graphViz, graph
}

func solveNoSplitDelta(ematrix EMatrix, parLambda float64, lossKind SplitLoss, depth int) *mat.Dense {
	delta := mat.NewDense(depth, 1, nil)
	h, d := ematrix.FeaturesExtra.Dims()
	if h == 0 || d == 0 {
		return delta
	}

	if _, ok := lossKind.(MseLoss); !ok {
		return delta
	}

	var normal mat.Dense
	normal.Mul(ematrix.FeaturesExtra.T(), ematrix.FeaturesExtra)
	for i := 0; i < d; i++ {
		normal.Set(i, i, normal.At(i, i)+parLambda)
	}

	var rhs mat.Dense
	rhs.Mul(ematrix.FeaturesExtra.T(), ematrix.Target)

	var solve mat.Dense
	solve.CloneFrom(&rhs)
	solve.Solve(&normal, &rhs)
	delta.Copy(&solve)
	return delta
}
