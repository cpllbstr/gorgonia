package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"gorgonia.org/gorgonia"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var (
	imgWidth       = 416
	imgHeight      = 416
	channels       = 3
	boxes          = 3
	leakyCoef      = 0.1
	cocoClasses    = []string{"person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"}
	scoreThreshold = float32(0.8)
	iouThreshold   = float32(0.3)
)

func main() {
	modeStr := flag.String("mode", "detector", "Choose the mode: detector/training")
	imagePathStr := flag.String("image", "./data/dog_416x416.jpg", "Path to image file for 'detector' mode")
	weightsPath := flag.String("weights", "./data/yolov3-tiny.weights", "Path to weights file")
	cfgPathStr := flag.String("cfg", "./data/yolov3-tiny.cfg", "Path to net configuration file")
	trainingFolder := flag.String("train", "./data/test_yolo_op", "Path to folder with labeled data")
	imgDir := flag.String("imdir", "./data/test_yolo_op", "Path to folder with labeled data")
	flag.Parse()

	g := G.NewGraph()
	input := G.NewTensor(g, tensor.Float32, 4, G.WithShape(1, channels, imgWidth, imgHeight), G.WithName("input"))
	model, err := NewYoloV3Tiny(g, input, len(cocoClasses), boxes, leakyCoef, *cfgPathStr, *weightsPath)
	if err != nil {
		fmt.Printf("Can't prepare YOLOv3 network due the error: %s\n", err.Error())
		return
	}
	model.Print()

	switch *modeStr {
	case "detector":

		imgf32, err := GetFloat32Image(*imagePathStr, imgHeight, imgWidth)
		if err != nil {
			fmt.Printf("Can't read []float32 from image due the error: %s\n", err.Error())
			return
		}
		image := tensor.New(tensor.WithShape(1, channels, imgHeight, imgWidth), tensor.Of(tensor.Float32), tensor.WithBacking(imgf32))
		err = G.Let(input, image)
		if err != nil {
			fmt.Printf("Can't let input = []float32 due the error: %s\n", err.Error())
			return
		}

		tm := G.NewTapeMachine(g)
		defer tm.Close()

		// Feedforward
		st := time.Now()
		if err := tm.RunAll(); err != nil {
			fmt.Printf("Can't run tape machine due the error: %s\n", err.Error())
			return
		}
		fmt.Println("Feedforwarded in:", time.Since(st))

		// Postprocessing
		st = time.Now()
		dets, err := model.ProcessOutput(cocoClasses, scoreThreshold, iouThreshold)
		if err != nil {
			fmt.Printf("Can't do postprocessing due error: %s", err.Error())
			return
		}
		fmt.Println("Postprocessed in:", time.Since(st))

		fmt.Println("Detections:")
		for i := range dets {
			fmt.Println(dets[i])
		}

		tm.Reset()
		return
	case "training":
		// @todo
		labeledData, err := parseFolder(*trainingFolder)
		if err != nil {
			fmt.Printf("Can't prepare labeled data due the error: %s\n", err.Error())
			return
		}

		err = model.ActivateTrainingMode()
		if err != nil {
			fmt.Printf("Can't activate training mode due the error: %s\n", err.Error())
			return
		}

		solver := gorgonia.NewRMSPropSolver(gorgonia.WithLearnRate(0.00001))
		modelOut := model.GetOutput()
		concatOut, err := gorgonia.Concat(1, modelOut...)
		if err != nil {
			fmt.Printf("Can't concatenate YOLO layers outputs in Training mode due the error: %s\n", err.Error())
			return
		}
		costs, err := gorgonia.Sum(concatOut, 0, 1, 2)
		if err != nil {
			fmt.Printf("Can't evaluate costs in Training mode due the error: %s\n", err.Error())
			return
		}
		_, err = gorgonia.Grad(costs, model.learningNodes...)
		if err != nil {
			fmt.Printf("Can't evaluate gradients in Training mode due the error: %s\n", err.Error())
			return
		}
		prog, locMap, err := gorgonia.Compile(g)
		if err != nil {
			fmt.Printf("Can't compile graph in Training mode due the error: %s\n", err.Error())
			return
		}

		tm := G.NewTapeMachine(g, gorgonia.WithPrecompiled(prog, locMap), gorgonia.BindDualValues(model.learningNodes...))
		defer tm.Close()
		i := 0
		fmt.Println("HELO!")
		for k := range labeledData {
			err = model.SetTarget(labeledData[k])
			if err != nil {
				fmt.Printf("Can't set []float32 as target due the error: %s\n", err.Error())
				return
			}
			imgf32, err := GetFloat32Image(fmt.Sprintf("%s/%s.jpg", *imgDir, k), imgHeight, imgWidth)
			if err != nil {
				fmt.Printf("Can't read []float32 from image due the error: %s\n", err.Error())
				return
			}

			image := tensor.New(tensor.WithShape(1, channels, imgHeight, imgWidth), tensor.Of(tensor.Float32), tensor.WithBacking(imgf32))
			err = G.Let(input, image)
			if err != nil {
				fmt.Printf("Can't let input = []float32 due the error: %s\n", err.Error())
				return
			}
			st := time.Now()
			if err := tm.RunAll(); err != nil {
				fmt.Printf("Can't run tape machine due the error: %s\n", err.Error())
				return
			}
			if i == 15 {
				solver = gorgonia.NewRMSPropSolver(gorgonia.WithLearnRate(0.000001))
			}
			if i == 150 {
				solver = gorgonia.NewRMSPropSolver(gorgonia.WithLearnRate(0.0000001))
			}
			fmt.Printf("Training iteration #%d done in: %v\n", i, time.Since(st))
			fmt.Printf("\tCurrent costs are: %v\n", costs.Value())

			err = solver.Step(gorgonia.NodesToValueGrads(model.learningNodes))
			if err != nil {
				fmt.Printf("Can't do solver.Step() in Training mode due the error: %s\n", err.Error())
			}
			i++

			tm.Reset()
		}

		return
	default:
		fmt.Printf("Mode %s is not supported!\nTry: training, detector", *modeStr)
		return
	}
}

func parseFolder(dir string) (map[string][]float32, error) {
	filesInfo, err := ioutil.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	targets := map[string][]float32{}
	for i := range filesInfo {
		sliceOfF32 := []float32{}
		fileInfo := filesInfo[i]
		// Parse only *.txt files
		if fileInfo.IsDir() || filepath.Ext(fileInfo.Name()) != ".txt" {
			continue
		}
		filePath := fmt.Sprintf("%s/%s", dir, fileInfo.Name())
		fileBytes, err := ioutil.ReadFile(filePath)
		if err != nil {
			return nil, err
		}
		fileContentAsArray := strings.Split(strings.ReplaceAll(string(fileBytes), "\n", " "), " ")
		for j := range fileContentAsArray {
			entity := fileContentAsArray[j]
			if entity == "" {
				continue
			}
			entityF32, err := strconv.ParseFloat(entity, 32)
			if err != nil {
				// Do we need return? May be just some warning...
				return nil, err
			}
			sliceOfF32 = append(sliceOfF32, float32(entityF32))
		}
		targets[strings.Split(fileInfo.Name(), ".")[0]] = sliceOfF32
	}

	if len(targets) == 0 {
		return nil, fmt.Errorf("Folder '%s' doesn't contain any *.txt files (annotation files for YOLO)", dir)
	}

	return targets, nil
}

func parseTargets(txtDir, imgDir string) (map[string][]float32, error) {
	imgexist := func(filename string) bool {
		info, err := os.Stat(imgDir + "/" + filename + ".jpg")
		if os.IsNotExist(err) {
			return false
		}
		return !info.IsDir()
	}
	filesInfo, err := ioutil.ReadDir(txtDir)
	if err != nil {
		return nil, err
	}

	targets := map[string][]float32{}
	for i := range filesInfo {
		sliceOfF32 := []float32{}
		fileInfo := filesInfo[i]
		// Parse only *.txt files
		if fileInfo.IsDir() || filepath.Ext(fileInfo.Name()) != ".txt" {
			continue
		}
		img := fmt.Sprintf("%s/%s.jpg", imgDir, strings.Split(fileInfo.Name(), ".")[0])
		if !imgexist(img) {
			fmt.Printf("!\tImage %s does not exist!\n", img)
			continue
		}
		filePath := fmt.Sprintf("%s/%s", txtDir, fileInfo.Name())
		fileBytes, err := ioutil.ReadFile(filePath)
		if err != nil {
			return nil, err
		}
		fileContentAsArray := strings.Split(strings.ReplaceAll(string(fileBytes), "\n", " "), " ")
		for j := range fileContentAsArray {
			entity := fileContentAsArray[j]
			if entity == "" {
				continue
			}
			entityF32, err := strconv.ParseFloat(entity, 32)
			if err != nil {
				// Do we need return? May be just some warning...
				return nil, err
			}
			sliceOfF32 = append(sliceOfF32, float32(entityF32))
		}
		targets[strings.Split(fileInfo.Name(), ".")[0]] = sliceOfF32
	}

	if len(targets) == 0 {
		return nil, fmt.Errorf("Folder '%s' doesn't contain any *.txt files (annotation files for YOLO) or folder %s doesn't contain any images", txtDir, imgDir)
	}

	return targets, nil
}
