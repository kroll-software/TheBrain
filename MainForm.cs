using System;
using System.Linq;
using System.Drawing;
using System.Collections.Generic;
using KS.Foundation;
using OpenTK.Windowing.Common;
using OpenTK.Windowing.GraphicsLibraryFramework;
using SummerGUI;
using SummerGUI.Charting;
using TheBrain.CudaSNN;

namespace TheBrain
{
	public class MainForm : ApplicationWindow
	{
        ImageList ImlToolbar;

		public ToolBarCircleButton m_CmdPrevious;
		public ToolBarCircleButton m_CmdNext;
		public ToolBarSeparator m_Separator1;
		public ToolBarImageButton m_CmdDice;
		public ToolBarImageButton m_CmdRun;
		public ToolBarImageButton m_CmdScore;		

		public PlotterContainer m_GraphPlotter;      

		public SnnModel SNN {get; private set; }

		public SnnWidget SNNWidget {get; private set; }

		public MainForm () : base("TheBrain (Kroll-Software)", 800, 600)
		{
		}

		protected override void OnLoad (EventArgs e)
		{
			base.OnLoad (e);

			LeftSideBarVisible = false;

			this.MenuPanel.MenuBarVisible = false;

			ImlToolbar = ImageList.FromFolder (this, "Assets/ToolBar", "toolbarImages");
			MenuPanel.ToolBar.Images = ImlToolbar;

			MenuPanel.ToolBar.Clear ();

			m_CmdPrevious = new ToolBarCircleButton ("previous", "Previous", "Previous_30px.png");
			MenuPanel.ToolBar.AddChild (m_CmdPrevious);

			m_CmdNext = new ToolBarCircleButton ("next", "Next", "Next_30px.png");
			MenuPanel.ToolBar.AddChild (m_CmdNext);

			m_Separator1 = new ToolBarSeparator ("separator1");
			MenuPanel.ToolBar.AddChild (m_Separator1);

			m_CmdDice = new  ToolBarImageButton ("cmdDice", "Dice", "Dice-2_30px.png");
			MenuPanel.ToolBar.AddChild (m_CmdDice);

			m_CmdDice.Click += (sender, args) => BuildSnnModel();

			//m_CmdSNN  = new ToolBarButton ("cmdSNN", String.Empty, "Assets\\ToolBar\\Brain-3_30px.png".FixedExpandedPath ());
			m_CmdRun = new ToolBarImageButton ("cmdRun", "Run", "Running_30px.png");
			m_CmdRun.IsToggleButton = true;
			MenuPanel.ToolBar.AddChild (m_CmdRun);

			m_CmdRun.CheckedChanged += delegate {
				this.LogInformation("m_CmdRun.Checked: {0}", m_CmdRun.Checked);
			};

			m_CmdScore = new ToolBarImageButton ("cmdScore", "Score", "Trophy_30px.png");
			MenuPanel.ToolBar.AddChild (m_CmdScore);

			TabMain.AdTabPage ("snn", "SNN");
			TabMain.AdTabPage ("midi", "MIDI");
			TabMain.AdTabPage ("plotter", "Plotter");

			SNNWidget = new SnnWidget ("Brain");
			TabMain.TabPages ["snn"].AddChild (SNNWidget);			

			m_GraphPlotter = new PlotterContainer ("graph2d");
			m_GraphPlotter.Plotter.EditMode = Graph2DPlotter.PlotterEditModes.editNone;
			m_GraphPlotter.Plotter.StringFormatY = "F5";
			//m_GraphPlotter.Plotter.yRange = 1d;
			TabMain.TabPages ["plotter"].AddChild (m_GraphPlotter);				

			ShowStatus();
		}        

		public void BuildSnnModel()
		{
			ShowStatus("Building SNN-Model ...", false);
			SNN = new SnnModel("test");			

			SNN.BuildNetwork(new BrainConfiguration{				
				NumInputClasses = 16384,
				NumInputClassNeurons = 1,
				NumHiddenLayers = 1,
				NeuronsPerHiddenLayer = 262144,
				HiddenLayerMaxSynapses = 512,
				NumOutputClasses = 16384,
				NumOutputClassNeurons = 1,
				OutputLayerMaxSynapses = 1024
			});

			SNNWidget.SNN = SNN;
			
			ShowStatus("Training ...", false);
			
			Task.Run(() => {
				BookTrainer trainer = new BookTrainer(SNN, 16384);
				trainer.TrainOnDirectory("/home/detlef/The_Brain_Books/");
			});
		}

		protected override void OnKeyDown (KeyboardKeyEventArgs e)
		{
			if (e.Key == Keys.F11)
				this.ToggleFullScreen ();
			else
				base.OnKeyDown (e);
		}

		protected override void Dispose (bool manual)
		{
			if (manual) {
				ImlToolbar?.Dispose ();
			}
			base.Dispose (manual);
		}
	}
}

