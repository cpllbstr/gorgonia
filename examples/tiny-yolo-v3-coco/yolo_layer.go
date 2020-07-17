package main

import (
	"fmt"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type yoloLayer struct {
	masks          []int
	anchors        [][2]int
	flattenAhcnors []int
	inputSize      int
	classesNum     int
}

func (l *yoloLayer) String() string {
	str := "YOLO layer: "
	for m := range l.masks {
		str += fmt.Sprintf("Mask->%[1]d Anchors->[%[2]d, %[3]d]", l.masks[m], l.anchors[m][0], l.anchors[m][1])
		if m != len(l.masks)-1 {
			str += "\t|\t"
		}
	}
	return str
}

func (l *yoloLayer) Type() string {
	return "yolo"
}

func (l *yoloLayer) ToNode(g *gorgonia.ExprGraph, input ...*gorgonia.Node) (*gorgonia.Node, error) {
	inputN := input[0]
	if len(inputN.Shape()) == 0 {
		return nil, fmt.Errorf("Input shape for YOLO layer is nil")
	}
	preparedTensor := gorgonia.NewTensor(g, tensor.Float64, 4, gorgonia.WithShape(inputN.Shape()...), gorgonia.WithName("yolo"), gorgonia.WithInit(gorgonia.Zeroes()))
	fmt.Println(l.flattenAhcnors, l.inputSize, l.classesNum)
	yoloNode, err := gorgonia.YOLOv3(preparedTensor, l.flattenAhcnors, l.inputSize, l.classesNum)
	if err != nil {
		return nil, errors.Wrap(err, "Can't prepare YOLOv3 operation")
	}
	return yoloNode, nil
}
