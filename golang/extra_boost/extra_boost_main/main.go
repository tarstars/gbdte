package main

import (
	"encoding/json"
	"flag"
	"github.com/goccy/go-graphviz"
	"github.com/sbinet/npyio"
	"github.com/tarstars/extra_bridged_boosting/golang/extra_boost/ebl"
	"gonum.org/v1/gonum/mat"
	"log"
	"math"
	"os"
	"path"
	"runtime"
	"runtime/pprof"
)

func decodeConfig(srcConfig string, out interface{}) {
	file, err := os.Open(srcConfig)
	ebl.HandleError(err)
	defer func() { ebl.HandleError(file.Close()) }()

	decoder := json.NewDecoder(file)
	ebl.HandleError(decoder.Decode(out))
}

func trainModel() {
	baseDir := "/home/tass/database/app_in_the_air/demand_predictions/current_data_set"
	log.Println("load train")
	ematrix_train := ebl.ReadEMatrix(
		path.Join(baseDir, "inter_train_stable.npy"), // "inter_train_debug_incrementing.npy"
		path.Join(baseDir, "extra_train.npy"),
		path.Join(baseDir, "target_train.npy"),
	)

	log.Println("load test")
	ematrix_test := ebl.ReadEMatrix(
		path.Join(baseDir, "inter_test_stable.npy"), // "inter_test_debug_incrementing.npy"
		path.Join(baseDir, "extra_test.npy"),
		path.Join(baseDir, "target_test.npy"),
	)

	clf := ebl.NewEBooster(ebl.EBoosterParams{
		Matrix:         ematrix_train,
		NStages:        2000,
		RegLambda:      1e-4,
		MaxDepth:       6,
		LearningRate:   0.3,
		LossKind:       ebl.MseLoss{},
		PrintMessages:  []ebl.EMatrix{ematrix_train, ematrix_test},
		ThreadsNum:     40,
		UnbalancedLoss: 0,
	})

	graphViz, graph := clf.Trees[0].DrawGraph()
	ebl.HandleError(graphViz.RenderFilename(graph, graphviz.SVG, "tree_00.svg"))

	clf.Save(path.Join(baseDir, "golang_model_1000.ebm"))
}

type TestConfig struct {
	Description        string `json:"description"`
	FileNameTestInter  string `json:"filename_test_inter"`
	FileNameTestExtra  string `json:"filename_test_extra"`
	FileNameTestTarget string `json:"filename_test_target"`
}

type TrainConfig struct {
	FileNameTrainInter  string       `json:"filename_train_inter"`
	FileNameTrainExtra  string       `json:"filename_train_extra"`
	FileNameTrainTarget string       `json:"filename_train_target"`
	Tests               []TestConfig `json:"tests"`
	FileNameModel       string       `json:"filename_model"`
	NStages             int          `json:"n_stages"`
	RegLambda           float64      `json:"reg_lambda"`
	MaxDepth            int          `json:"max_depth"`
	LearningRate        float64      `json:"learning_rate"`
	ThreadsNum          int          `json:"threads_num"`
	UnbalancedLoss      float64      `json:"unbalanced_loss"`
}

func train(srcConfig string) {
	var trainConfig TrainConfig
	decodeConfig(srcConfig, &trainConfig)

	ematrixTrain := ebl.ReadEMatrix(
		trainConfig.FileNameTrainInter,
		trainConfig.FileNameTrainExtra,
		trainConfig.FileNameTrainTarget,
	)

	var ematrixTests []ebl.EMatrix
	for _, testConfig := range trainConfig.Tests {
		ematrix := ebl.ReadEMatrix(
			testConfig.FileNameTestInter,
			testConfig.FileNameTestExtra,
			testConfig.FileNameTestTarget,
		)
		ematrix.SetDescription(testConfig.Description)
		ematrixTests = append(ematrixTests, ematrix)
	}

	clf := ebl.NewEBooster(ebl.EBoosterParams{
		Matrix:         ematrixTrain,
		NStages:        trainConfig.NStages,
		RegLambda:      trainConfig.RegLambda,
		MaxDepth:       trainConfig.MaxDepth,
		LearningRate:   trainConfig.LearningRate,
		LossKind:       ebl.MseLoss{},
		PrintMessages:  ematrixTests,
		ThreadsNum:     trainConfig.ThreadsNum,
		UnbalancedLoss: trainConfig.UnbalancedLoss,
	})

	clf.Save(trainConfig.FileNameModel)
}

type PredictConfig struct {
	DataInterFileName  string `json:"filename_feature_inter"`
	DataExtraFileName  string `json:"filename_feature_extra"`
	ModelFileName      string `json:"filename_model"`
	PredictionFileName string `json:"filename_target"`
	TreesNumber        int    `json:"trees_number"`
}

func predict(srcConfig string) {
	var predictConfig PredictConfig
	decodeConfig(srcConfig, &predictConfig)

	FeaturesInter := ebl.ReadNpy(predictConfig.DataInterFileName)
	FeaturesExtra := ebl.ReadNpy(predictConfig.DataExtraFileName)

	//interH, interW := FeaturesInter.Dims()
	//extraH, extraW := FeaturesExtra.Dims()

	//log.Println("inter dims = ", interH, " ", interW)
	//log.Println("inter dims = ", extraH, " ", extraW)

	clf := ebl.LoadModel(predictConfig.ModelFileName)

	var optionalTreeNumber *int
	if predictConfig.TreesNumber != 0 {
		optionalTreeNumber = &predictConfig.TreesNumber
	}

	prediction := clf.PredictValue(FeaturesInter, FeaturesExtra, optionalTreeNumber)
	dst, err := os.Create(predictConfig.PredictionFileName)
	ebl.HandleError(err)
	ebl.HandleError(npyio.Write(dst, prediction))
}

type LcurveConfig struct {
	DataInterFileName     string `json:"filename_feature_inter"`
	DataExtraFileName     string `json:"filename_feature_extra"`
	ModelFileName         string `json:"filename_model"`
	PredictionFileName    string `json:"filename_target"`
	LearningCurveFileName string `json:"filename_learning_curve"`
}

func lcurve(srcConfig string) {
	var lcurveConfig LcurveConfig
	decodeConfig(srcConfig, &lcurveConfig)

	FeaturesInter := ebl.ReadNpy(lcurveConfig.DataInterFileName)
	FeaturesExtra := ebl.ReadNpy(lcurveConfig.DataExtraFileName)
	Target := ebl.ReadNpy(lcurveConfig.PredictionFileName)

	clf := ebl.LoadModel(lcurveConfig.ModelFileName)

	learningCurve := mat.NewDense(len(clf.Trees), 1, nil)

	var prediction *mat.Dense

	for currentTreeNumber, currentTree := range clf.Trees {
		if prediction == nil {
			prediction = currentTree.PredictValue(FeaturesInter, FeaturesExtra)
		} else {
			prediction.Add(prediction, currentTree.PredictValue(FeaturesInter, FeaturesExtra))
		}
		currentLoss := 0.0

		predictionH, _ := prediction.Dims()
		targetH, _ := Target.Dims()
		if predictionH != targetH {
			log.Panic("prediction and target have different shapes")
		}

		h := ebl.Height(prediction)
		for ind := 0; ind < h; ind++ {
			d := prediction.At(ind, 0) - Target.At(ind, 0)
			currentLoss += d * d
		}
		currentLoss = math.Sqrt(currentLoss / float64(h))

		learningCurve.Set(currentTreeNumber, 0, currentLoss)
	}

	dst, err := os.Create(lcurveConfig.LearningCurveFileName)
	ebl.HandleError(err)
	ebl.HandleError(npyio.Write(dst, learningCurve))
}

type GraphConfig struct {
	ModelFileName     string `json:"filename_model"`
	FigureType        string `json:"figure_type"`
	PicturesDirectory string `json:"pictures_directory"`
	DumpPrefix        string `json:"dump_prefix"`
}

func graph(srcConfig string) {
	var graphConfig GraphConfig
	decodeConfig(srcConfig, &graphConfig)

	clf := ebl.LoadModel(graphConfig.ModelFileName)
	clf.RenderTrees(graphConfig.DumpPrefix, graphConfig.FigureType, graphConfig.PicturesDirectory)
}

type ModelLearningCurvesConfig struct {
	PathToModel            string `json:"path_to_model"`
	FilenameLearningCurves string `json:"filename_learning_curves"`
}

func getLearningCurves(srcConfig string) {
	var modelLearningCurves ModelLearningCurvesConfig
	decodeConfig(srcConfig, &modelLearningCurves)

	clf := ebl.LoadModel(modelLearningCurves.PathToModel)
	clf.DumpLearningCurves(modelLearningCurves.FilenameLearningCurves)
}

func main() {
	runMode := flag.String("mode", "train", "you can select either 'train', 'graph', 'predict' or 'lcurve' modes")
	config := flag.String("config", "extra_config.json", "a config file for the run of the program")
	memprofile := flag.String("memprofile", "", "write memory profile to `file`")

	flag.Parse()

	map[string]func(string){
		"train":               train,
		"predict":             predict,
		"graph":               graph,
		"lcurve":              lcurve,
		"get_learning_curves": getLearningCurves,
	}[*runMode](*config)

	if *memprofile != "" {
		f, err := os.Create(*memprofile)
		ebl.HandleError(err)
		defer func() { ebl.HandleError(f.Close()) }()
		runtime.GC()
		if err := pprof.WriteHeapProfile(f); err != nil {
			log.Fatal("could not write memory profile: ", err)
		}
	}
}
