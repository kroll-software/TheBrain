using System;
using System.IO;
using KS.Foundation;
using System.Drawing;
using SummerGUI;
using TheBrain.CudaSNN;

namespace TheBrain;

public class SnnWidget : Widget
{
    public SnnModel SNN { get; set; }
    public BookTrainer Trainer { get; set; }

    SummerGUI.Brush[] Brushes;
    SummerGUI.Brush FontBrush;

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

    public IGUIFont Font {get; private set;}

    public SnnWidget(string name) 
        : base(name, Docking.Fill, null)
    {
        Brushes = new SummerGUI.Brush[] {		
				new SummerGUI.SolidBrush(SolarizedColors.Silver),
				new SummerGUI.SolidBrush(SolarizedColors.Yellow),
				new SummerGUI.SolidBrush(SolarizedColors.Green),
				new SummerGUI.SolidBrush(SolarizedColors.Cyan),
				new SummerGUI.SolidBrush(SolarizedColors.Orange),
				new SummerGUI.SolidBrush(SolarizedColors.Red),
				new SummerGUI.SolidBrush(SolarizedColors.Blue),
				new SummerGUI.SolidBrush(SolarizedColors.Violet),
				new SummerGUI.SolidBrush(SolarizedColors.Base2),
				new SummerGUI.SolidBrush(SolarizedColors.Base3),
				new SummerGUI.SolidBrush(BackColor),
                new SummerGUI.SolidBrush(Color.Black),
				//new LinearGradientBrush (SolarizedColors.Base02, SolarizedColors.Base03, GradientDirections.Vertical),
				new SummerGUI.SolidBrush(SolarizedColors.Silver)
			};

            Font = FontManager.Manager.DefaultFont;
            FontBrush = new SummerGUI.SolidBrush(Theme.Colors.Base00);

            BackColor = Color.Black;
    }

    public override void OnPaint(IGUIContext ctx, RectangleF bounds)
    {
        base.OnPaint(ctx, bounds);
        if (SNN == null || Trainer == null)
            return;

        var data = SNN.GetNeuronPotentialsForDrawing();

        float scaleX = bounds.Width / 640f;
        float scaleY = bounds.Height / 640f;
        float pointSize = 3f;
        
        for (int i = 0; i < data.Length; i += 25)
        {
            var n = data[i];
            SummerGUI.Brush brush;
            switch(n.State)
            {                
                case 1:
                    if (n.Type == 0)
                        brush = Brushes[8];
                    else
                        brush = Brushes[6];
                    break;
                case 2:
                    if (n.Type == 0)
                        brush = Brushes[7];
                    else
                        brush = Brushes[4];
                    break;
                default:
                    if (n.Type == 0)
                        brush = Brushes[11];
                    else
                        brush = Brushes[11];
                    break;
            }            
            
            float drawX = bounds.Left + (n.PosX * scaleX);
            float drawY = bounds.Top + (n.PosY * scaleY);            

            RectangleF r = new RectangleF(drawX, drawY, pointSize, pointSize);
            ctx.FillRectangle(brush, r);
        }

        int synapseCount = SNN.GetDynamicSynapseCount();
        RectangleF rstatus = new RectangleF(bounds.Left + 12, bounds.Top + 12, bounds.Width - 24, Font.LineHeight);
        ctx.DrawString($"Iter: {SNN.Iteration:N0} | Synapses: {synapseCount:N0} | Book: {Path.GetFileName(Trainer.BookFile)}", Font, FontBrush, rstatus, FontFormat.DefaultSingleLine);

        Invalidate();
    }
}
