// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package catsdogs

import (
	"github.com/emer/leabra/v2/leabra"
)

// LayerParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var LayerParams = leabra.LayerSheets{
	"Base": {
		{Sel: "Layer", Doc: "generic params for all layers: lower gain, slower, soft clamp",
			Set: func(ly *leabra.LayerParams) {
				ly.Inhib.ActAvg.Init = 0.25
				ly.Inhib.ActAvg.Fixed = true
				ly.Inhib.Layer.FBTau = 3 // this is key for smoothing bumps
				ly.Act.Clamp.Hard = false
				ly.Act.Clamp.Gain = 1
				ly.Act.XX1.Gain = 40 // more graded -- key
				ly.Act.Dt.VmTau = 4 // a bit slower -- not as effective as FBTau
				ly.Act.Gbar.L = 0.1 // needs lower leak
			}},
		{Sel: ".Id", Doc: "specific inhibition for identity, name",
			Set: func(ly *leabra.LayerParams) {
				ly.Inhib.Layer.Gi = 4.0
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
