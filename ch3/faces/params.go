// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package faces

import (
	"github.com/emer/leabra/v2/leabra"
)

// LayerParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var LayerParams = leabra.LayerSheets{
	"Base": {
		{Sel: "Layer", Doc: "fix expected activity levels, reduce leak",
			Set: func(ly *leabra.LayerParams) {
				ly.Inhib.ActAvg.Init = 0.15
				ly.Inhib.ActAvg.Fixed = true
				ly.Act.Gbar.L = 0.1 // needs lower leak
			}},
		{Sel: "#Input", Doc: "specific inhibition",
			Set: func(ly *leabra.LayerParams) {
				ly.Inhib.Layer.Gi = 2.0
				ly.Act.Clamp.Hard = false
				ly.Act.Clamp.Gain = 0.2
			}},
		{Sel: "#Identity", Doc: "specific inhibition",
			Set: func(ly *leabra.LayerParams) {
				ly.Inhib.Layer.Gi = 3.6
			}},
		{Sel: "#Gender", Doc: "specific inhibition",
			Set: func(ly *leabra.LayerParams) {
				ly.Inhib.Layer.Gi = 1.6
			}},
		{Sel: "#Emotion", Doc: "specific inhibition",
			Set: func(ly *leabra.LayerParams) {
				ly.Inhib.Layer.Gi = 1.3
			}},
	},
}

// PathParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var PathParams = leabra.PathSheets{
	"Base": {
		{Sel: "Path", Doc: "basic path params",
			Set: func(pt *leabra.PathParams) {
				pt.Learn.Learn = false
			}},
	},
}
