package main

import (
	"fmt"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
)

type yoloLayer struct {
	treshold   float32
	mask       []int
	anchors    []int
	inputSize  int
	classesNum int
}

func (l *yoloLayer) String() string {
	str := "YOLO layer: "
	for _, m := range l.mask {
		str += fmt.Sprintf("Mask->%[1]d Anchors->[%[2]d, %[3]d]", m, l.anchors[2*m], l.anchors[2*m+1])
		if m != len(l.mask)-1 {
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
	yoloNode, err := gorgonia.YOLOv3(inputN, l.anchors, l.mask, l.inputSize, l.classesNum, l.treshold)
	if err != nil {
		return nil, errors.Wrap(err, "Can't prepare YOLOv3 operation")
	}
	return yoloNode, nil
}
