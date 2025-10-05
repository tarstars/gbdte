package ebl

import (
	"github.com/goccy/go-graphviz"
	"log"
	"path"
	"testing"
)

func TestTrainModel(_ *testing.T) {
	baseDir := "/home/tass/database/app_in_the_air/demand_predictions/debug_memory"
	log.Println("load train")
	ematrix_train := ReadEMatrix(
		path.Join(baseDir, "known_present_inter.npy"), // "inter_train_debug_incrementing.npy"
		path.Join(baseDir, "known_present_extra.npy"),
		path.Join(baseDir, "known_present_target.npy"),
	)

	log.Println("load test")
	ematrix_test := ReadEMatrix(
		path.Join(baseDir, "known_future_inter.npy"), // "inter_test_debug_incrementing.npy"
		path.Join(baseDir, "known_future_extra.npy"),
		path.Join(baseDir, "known_future_target.npy"),
	)

	clf := NewEBooster(EBoosterParams{
		Matrix:         ematrix_train,
		NStages:        20,
		RegLambda:      1e-4,
		MaxDepth:       6,
		LearningRate:   0.3,
		LossKind:       MseLoss{},
		PrintMessages:  []EMatrix{ematrix_train, ematrix_test},
		ThreadsNum:     5,
		UnbalancedLoss: 0,
	})

	graphViz, graph := clf.Trees[0].DrawGraph()
	HandleError(graphViz.RenderFilename(graph, graphviz.SVG, "tree_00.svg"))

	clf.Save(path.Join(baseDir, "golang_model_1000.ebm"))
}
