using System;
using KS.Foundation;
using System.Drawing;
using SummerGUI;
using TheBrain.CudaSNN;

namespace TheBrain;

public class SnnWidget : Widget
{
    public SnnModel SNN { get; set; }

    SummerGUI.Brush[] Brushes;

    private enum NeuronColors
    {	
        Empty,
        Yellow,
        Green,
        Cyan,
        Orange,
        Red,
        Blue,
        Violet,
        ActiveBack,
        InactiveBack,
        Gray,
        Silver
    }

    public SnnWidget(string name) 
        : base(name, Docking.Fill, null)
    {
        Brushes = new SummerGUI.Brush[] {		
				new SummerGUI.SolidBrush(Color.FromArgb(245, SolarizedColors.Silver)),
				new SummerGUI.SolidBrush(Color.FromArgb(245, SolarizedColors.Yellow)),
				new SummerGUI.SolidBrush(Color.FromArgb(240, SolarizedColors.Green)),
				new SummerGUI.SolidBrush(Color.FromArgb(240, SolarizedColors.Cyan)),
				new SummerGUI.SolidBrush(Color.FromArgb(240, SolarizedColors.Orange)),
				new SummerGUI.SolidBrush(Color.FromArgb(240, SolarizedColors.Red)),
				new SummerGUI.SolidBrush(Color.FromArgb(240, SolarizedColors.Blue)),
				new SummerGUI.SolidBrush(Color.FromArgb(170, SolarizedColors.Violet)),
				new SummerGUI.SolidBrush(SolarizedColors.Base2),
				new SummerGUI.SolidBrush(SolarizedColors.Base3),
				new SummerGUI.SolidBrush(BackColor),
				//new LinearGradientBrush (SolarizedColors.Base02, SolarizedColors.Base03, GradientDirections.Vertical),
				new SummerGUI.SolidBrush(SolarizedColors.Silver)
			};
    }

    public override void OnPaint(IGUIContext ctx, RectangleF bounds)
    {
        base.OnPaint(ctx, bounds);
        if (SNN == null)
            return;

        var data = SNN.GetNeuronPotentialsForDrawing();
        
        for (int i = 0; i < data.Length; i += 25)
        {
            var n = data[i];
            SummerGUI.Brush brush;
            switch(n.State)
            {                
                case 1:
                    brush = Brushes[5];
                    break;
                case 2:
                    brush = Brushes[7];
                    break;
                default:
                    brush = Brushes[6];
                    break;
            }
            
            RectangleF r = new RectangleF(n.PosX, n.PosY, 2f, 2f);
            ctx.FillRectangle(brush, r);
        }

        Invalidate(1);     
    }
}
