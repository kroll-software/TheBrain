using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using KS.Foundation;
using OpenTK;
using OpenTK.Input;
using OpenTK.Windowing.Common;
using OpenTK.Windowing.GraphicsLibraryFramework;
using SummerGUI;
using TheBrain.CudaSNN;

namespace TheBrain;

public class SnnEnsembleWidgetStyle : WidgetStyle
	{
		public SnnEnsembleWidgetStyle ()
			: base (SolarizedColors.Base01,
				SolarizedColors.Base02,
				System.Drawing.Color.Empty)
		{}
	}

	public class SnnEnsemble : SplitContainer
	{
		//public BrainToolBar ToolBar { get; protected set; }
		//public BrainControl BrainControl { get; protected set; }
		//public BrainPerformanceBar BrainCharts { get; protected set; }
		public SnnModel Brain { get; protected set; }

		public SnnEnsemble (string name)
			: base (name, SplitOrientation.Vertical, 0.75f)
		{						
			//BrainControl = new BrainControl ("brain");
			//this.Panel1.AddChild (BrainControl);

			//BrainCharts = new BrainPerformanceBar ("brainperform");
			//this.Panel2.AddChild (BrainCharts);

			this.Splitter.Distance = -220;
			this.FixedPanel = SplitterFixedPanel.Panel2;			
		}

        public override void Initialize()
        {
            base.Initialize();
        }
        
		public override bool OnKeyDown (KeyboardKeyEventArgs e)
		{
			if (base.OnKeyDown (e))
				return true;

			switch (e.Key) {
			case Keys.F1:
				Panel2Collapsed = !Panel2Collapsed;
				Invalidate ();
				return true;
			}

			return false;
		}

		public void InitBrain(BrainConfiguration configuration)
		{
			Brain = new SnnModel("TheBrain");
			Brain.BuildNetwork (configuration);

			//BrainControl.Brain = Brain;
			//BrainCharts.Brain = Brain;
		}

		//public event EventHandler BrainDeserialized;

		public void LoadBrain(string filename)
		{							
		}

		public void Run()
		{
			//Brain.Run ();
			//Brain.BrainMode = Brain.BrainModes.Dreaming;
		}

		public void Stop()
		{
			//Brain.Cancel ();
		}

		public void ClearBrain()
		{
			//Brain.Clear ();
		}
					
		protected override void CleanupManagedResources ()
		{
			//if (Brain != null)
			//	Brain.Cancel ();			
			base.CleanupManagedResources ();
		}
	}