package main

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// YoloV3Tiny YoloV3 tiny architecture
type YoloV3Tiny struct {
	g *gorgonia.ExprGraph

	out *gorgonia.Node

	biases  map[string][]float32
	gammas  map[string][]float32
	means   map[string][]float32
	vars    map[string][]float32
	kernels map[string][]float32
}

// type layer struct {
// 	name    string
// 	shape   tensor.Shape
// 	biases  []float32
// 	gammas  []float32
// 	means   []float32
// 	vars    []float32
// 	kernels []float32
// }

// NewYoloV3Tiny Create new tiny YOLO v3
func NewYoloV3Tiny(g *gorgonia.ExprGraph, input *gorgonia.Node, classesNumber, boxesPerCell int, leakyCoef float64, cfgFile, weightsFile string) (*YoloV3Tiny, error) {
	inputS := input.Shape()
	if len(inputS) < 4 {
		return nil, fmt.Errorf("Input for YOLOv3 should contain infromation about 4 dimensions")
	}

	buildingBlocks, err := ParseConfiguration(cfgFile)
	if err != nil {
		return nil, errors.Wrap(err, "Can't read darknet configuration")
	}

	weightsData, err := ParseWeights(weightsFile)
	if err != nil {
		return nil, errors.Wrap(err, "Can't read darknet weights")
	}

	fmt.Println("Loading network...")
	layers := []*layerN{}
	outputFilters := []int{}
	prevFilters := 3

	networkNodes := []*gorgonia.Node{}

	blocks := buildingBlocks[1:]
	for i := range blocks {
		block := blocks[i]
		filtersIdx := 0
		layerType, ok := block["type"]
		if ok {
			switch layerType {
			case "convolutional":
				filters := 0
				padding := 0
				kernelSize := 0
				stride := 0
				batchNormalize := 0
				bias := false
				activation := "activation"
				activation, ok := block["activation"]
				if !ok {
					fmt.Printf("No field 'activation' for convolution layer")
					continue
				}
				batchNormalizeStr, ok := block["batch_normalize"]
				batchNormalize, err := strconv.Atoi(batchNormalizeStr)
				if !ok || err != nil {
					batchNormalize = 0
					bias = true
				}
				filtersStr, ok := block["filters"]
				filters, err = strconv.Atoi(filtersStr)
				if !ok || err != nil {
					fmt.Printf("Wrong or empty 'filters' parameter for convolution layer: %s\n", err.Error())
					continue
				}
				paddingStr, ok := block["pad"]
				padding, err = strconv.Atoi(paddingStr)
				if !ok || err != nil {
					fmt.Printf("Wrong or empty 'pad' parameter for convolution layer: %s\n", err.Error())
					continue
				}
				kernelSizeStr, ok := block["size"]
				kernelSize, err = strconv.Atoi(kernelSizeStr)
				if !ok || err != nil {
					fmt.Printf("Wrong or empty 'size' parameter for convolution layer: %s\n", err.Error())
					continue
				}
				pad := 0
				if padding != 0 {
					pad = (kernelSize - 1) / 2
				}
				strideStr, ok := block["stride"]
				stride, err = strconv.Atoi(strideStr)
				if !ok || err != nil {
					fmt.Printf("Wrong or empty 'stride' parameter for convolution layer: %s\n", err.Error())
					continue
				}

				ll := &convLayer{
					filters:        filters,
					padding:        pad,
					kernelSize:     kernelSize,
					stride:         stride,
					activation:     activation,
					batchNormalize: batchNormalize,
					bias:           bias,
				}

				// ll.kernels = gorgonia.NewTensor(g, tensor.Float32, 4, gorgonia.WithShape(filters, prevFilters, kernelSize, kernelSize), gorgonia.WithName(fmt.Sprintf("conv_%d", i)))

				// if batchNormalize != 0 {
				// 	ll.beta = gorgonia.NewTensor(g, tensor.Float32, 1, gorgonia.WithShape(filters), gorgonia.WithName(fmt.Sprintf("beta_%d", i)))
				// 	ll.gamma = gorgonia.NewTensor(g, tensor.Float32, 1, gorgonia.WithShape(filters), gorgonia.WithName(fmt.Sprintf("gamma_%d", i)))
				// 	ll.vars = gorgonia.NewTensor(g, tensor.Float32, 1, gorgonia.WithShape(filters), gorgonia.WithName(fmt.Sprintf("vars_%d", i)))
				// 	ll.means = gorgonia.NewTensor(g, tensor.Float32, 1, gorgonia.WithShape(filters), gorgonia.WithName(fmt.Sprintf("means_%d", i)))
				// }

				var l layerN = ll
				convBlock, err := l.ToNode(g, input)
				if err != nil {
					fmt.Printf("\tError preparing Convolutional block: %s\n", err.Error())
				}
				networkNodes = append(networkNodes, convBlock)
				input = convBlock

				layers = append(layers, &l)
				fmt.Println(i, l, "\n", input.Shape())

				filtersIdx = filters
				break
			case "upsample":
				scale := 0
				scaleStr, ok := block["stride"]
				scale, err = strconv.Atoi(scaleStr)
				if !ok || err != nil {
					fmt.Printf("Wrong or empty 'stride' parameter for upsampling layer: %s\n", err.Error())
					continue
				}

				var l layerN = &upsampleLayer{
					scale: scale,
				}

				upsampleBlock, err := l.ToNode(g, input)
				if err != nil {
					fmt.Printf("\tError preparing Upsample block: %s\n", err.Error())
				}
				networkNodes = append(networkNodes, upsampleBlock)
				input = upsampleBlock

				layers = append(layers, &l)
				fmt.Println(i, l, "\n", input.Shape())

				// @todo upsample node

				filtersIdx = prevFilters
				break
			case "route":
				routeLayersStr, ok := block["layers"]
				if !ok {
					fmt.Printf("No field 'layers' for route layer")
					continue
				}
				layersSplit := strings.Split(routeLayersStr, ",")
				if len(layersSplit) < 1 {
					fmt.Printf("Something wrong with route layer. Check if it has one array item atleast")
					continue
				}
				for l := range layersSplit {
					layersSplit[l] = strings.TrimSpace(layersSplit[l])
				}
				start := 0
				end := 0
				start, err := strconv.Atoi(layersSplit[0])
				if err != nil {
					fmt.Printf("Each first element of 'layers' parameter for route layer should be an integer: %s\n", err.Error())
					continue
				}
				if len(layersSplit) > 1 {
					end, err = strconv.Atoi(layersSplit[1])
					if err != nil {
						fmt.Printf("Each second element of 'layers' parameter for route layer should be an integer: %s\n", err.Error())
						continue
					}
				}

				if start > 0 {
					start = start - i
				}
				if end > 0 {
					end = end - i
				}

				l := routeLayer{
					firstLayerIdx:  i + start,
					secondLayerIdx: -1,
				}
				if end < 0 {
					l.secondLayerIdx = i + end
					filtersIdx = outputFilters[i+start] + outputFilters[i+end]
				} else {
					filtersIdx = outputFilters[i+start]
				}

				var ll layerN = &l

				routeBlock, err := l.ToNode(g, networkNodes...)
				if err != nil {
					fmt.Printf("\tError preparing Route block: %s\n", err.Error())
				}
				networkNodes = append(networkNodes, routeBlock)
				input = routeBlock

				layers = append(layers, &ll)
				fmt.Println(i, ll, "\n", input.Shape())

				// @todo upsample node
				// @todo evaluate 'prevFilters'

				break
			case "yolo":
				maskStr, ok := block["mask"]
				if !ok {
					fmt.Printf("No field 'mask' for YOLO layer")
					continue
				}
				maskSplit := strings.Split(maskStr, ",")
				if len(maskSplit) < 1 {
					fmt.Printf("Something wrong with yolo layer. Check if it has one item in 'mask' array atleast")
					continue
				}
				masks := make([]int, len(maskSplit))
				for l := range maskSplit {
					maskSplit[l] = strings.TrimSpace(maskSplit[l])
					masks[l], err = strconv.Atoi(maskSplit[l])
					if err != nil {
						fmt.Printf("Each element of 'mask' parameter for yolo layer should be an integer: %s\n", err.Error())
					}
				}
				anchorsStr, ok := block["anchors"]
				if !ok {
					fmt.Printf("No field 'anchors' for YOLO layer")
					continue
				}
				anchorsSplit := strings.Split(anchorsStr, ",")
				if len(anchorsSplit) < 1 {
					fmt.Printf("Something wrong with yolo layer. Check if it has one item in 'anchors' array atleast")
					continue
				}
				if len(anchorsSplit)%2 != 0 {
					fmt.Printf("Number of elemnts in 'anchors' parameter for yolo layer should be divided exactly by 2 (even number)")
					continue
				}
				anchors := make([]int, len(anchorsSplit))
				for l := range anchorsSplit {
					anchorsSplit[l] = strings.TrimSpace(anchorsSplit[l])
					anchors[l], err = strconv.Atoi(anchorsSplit[l])
					if err != nil {
						fmt.Printf("Each element of 'anchors' parameter for yolo layer should be an integer: %s\n", err.Error())
					}
				}
				anchorsPairs := [][2]int{}
				for a := 0; a < len(anchors); a += 2 {
					anchorsPairs = append(anchorsPairs, [2]int{anchors[a], anchors[a+1]})
				}
				selectedAnchors := [][2]int{}
				for m := range masks {
					selectedAnchors = append(selectedAnchors, anchorsPairs[masks[m]])
				}

				var l layerN = &yoloLayer{
					masks:          masks,
					anchors:        selectedAnchors,
					flattenAhcnors: anchors,
					inputSize:      inputS[2],
					classesNum:     classesNumber,
				}
				yoloBlock, err := l.ToNode(g, input)
				if err != nil {
					fmt.Printf("\tError preparing YOLO block: %s\n", err.Error())
				}
				networkNodes = append(networkNodes, yoloBlock)
				input = yoloBlock

				layers = append(layers, &l)
				fmt.Println(i, l, "\n", input.Shape())

				// @todo detection node? or just flow?

				filtersIdx = prevFilters
				break
			case "maxpool":
				sizeStr, ok := block["size"]
				if !ok {
					fmt.Printf("No field 'size' for maxpooling layer")
					continue
				}
				size, err := strconv.Atoi(sizeStr)
				if err != nil {
					fmt.Printf("'size' parameter for maxpooling layer should be an integer: %s\n", err.Error())
					continue
				}
				strideStr, ok := block["stride"]
				if !ok {
					fmt.Printf("No field 'stride' for maxpooling layer")
					continue
				}
				stride, err := strconv.Atoi(strideStr)
				if err != nil {
					fmt.Printf("'size' parameter for maxpooling layer should be an integer: %s\n", err.Error())
					continue
				}

				var l layerN = &maxPoolingLayer{
					size:   size,
					stride: stride,
				}

				maxpoolingBlock, err := l.ToNode(g, input)
				if err != nil {
					fmt.Printf("\tError preparing Max-Pooling block: %s\n", err.Error())
				}
				networkNodes = append(networkNodes, maxpoolingBlock)
				input = maxpoolingBlock

				layers = append(layers, &l)
				fmt.Println(i, l, "\n", input.Shape())

				filtersIdx = prevFilters
				break
			default:
				fmt.Println("Impossible")
				break
			}
		}
		prevFilters = filtersIdx
		outputFilters = append(outputFilters, filtersIdx)
	}

	fmt.Println("Loading weights...")
	// lastIdx := 5 // skip first 5 values
	epsilon := float32(0.000001)

	ptr := 0
	for i := range layers {
		l := *layers[i]
		layerType := l.Type()
		// Ignore everything except convolutional layers
		if layerType == "convolutional" {
			layer := l.(*convLayer)
			var beta, gamma, means, vars, biases []float32
			fmt.Println("Loading weights: ", layer)
			if layer.batchNormalize > 0 && layer.outShape.TotalSize() > 0 {
				biasesNum := layer.bnOut.Shape()[1]

				beta = weightsData[ptr : ptr+biasesNum]
				err = gorgonia.Let(layer.beta, tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(biasesNum), tensor.WithBacking(beta)))
				ptr += biasesNum

				gamma = weightsData[ptr : ptr+biasesNum]
				err = gorgonia.Let(layer.gamma, tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(biasesNum), tensor.WithBacking(gamma)))
				ptr += biasesNum

				means = weightsData[ptr : ptr+biasesNum]
				err = gorgonia.Let(layer.means, tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(biasesNum), tensor.WithBacking(means)))
				ptr += biasesNum

				vars = weightsData[ptr : ptr+biasesNum]
				err = gorgonia.Let(layer.vars, tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(biasesNum), tensor.WithBacking(vars)))
				ptr += biasesNum
				if err != nil {
					panic(err)
				}
			} else {
				kernelNum := layer.kernels.Shape()[0]
				biases = weightsData[ptr : ptr+kernelNum]
				ptr += kernelNum
			}
			weightsNumel := layer.kernels.Shape().TotalSize()
			kernelW := weightsData[ptr : ptr+weightsNumel]

			if layer.batchNormalize > 0 && layer.outShape.TotalSize() > 0 {
				for i := 0; i < layer.kernels.Shape()[0]; i++ {
					scale := gamma[i] / float32(math.Sqrt(float64(vars[i]+epsilon)))

					beta[i] = beta[i] - means[i]*scale
					isize := layer.kernels.Shape()[1:4].TotalSize()
					for j := 0; j < isize; j++ {
						kernelW[isize*i+j] = kernelW[isize*i+j] * scale
					}
				}
			}
			err = gorgonia.Let(layer.kernels, tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(layer.kernels.Shape()...), tensor.WithBacking(kernelW)))
			err = gorgonia.Let(layer.biases, tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(layer.kernels.Shape()[0]), tensor.WithBacking(biases)))
			if err != nil {
				panic(err)
			}
			ptr += weightsNumel
		}
	}
	return nil, nil
}
