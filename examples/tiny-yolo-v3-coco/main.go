package main

import (
	"log"

	"gorgonia.org/gorgonia"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var (
	width     = 416
	height    = 416
	channels  = 3
	boxes     = 5
	classes   = 80
	leakyCoef = 0.1
	weights   = "./data/yolov3-tiny.weights"
	cfg       = "./data/yolov3-tiny.cfg"
)

func main() {
	g := G.NewGraph()

	input := gorgonia.NewTensor(g, tensor.Float32, 4, gorgonia.WithShape(1, channels, width, height), gorgonia.WithName("input"))

	model, err := NewYoloV3Tiny(g, input, classes, boxes, leakyCoef, cfg, weights)
	if err != nil {
		log.Fatalln(err)
	}
	img, err := GetFloat32Image("./data/dog_416x416.jpg")
	if err != nil {
		panic(err)
	}
	gorgonia.Let(input, tensor.New(tensor.WithShape(1, 3, 416, 416), tensor.WithBacking(img)))

	m := gorgonia.NewTapeMachine(g)

	// n := g.ByName("sconv_0")
	// fmt.Println(input.Value())

	err = m.RunAll()
	if err != nil {
		panic(err)
	}

	_ = model
}
