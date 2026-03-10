using System;
using System.Drawing;
using SummerGUI;

namespace KS.MI
{
	public class DayViewStyle : WidgetStyle
	{
		public DayViewStyle() : base(SolarizedColors.Base3, Color.Black, Color.Empty)
		{
		}
	}

	public static class Theme
	{
		static Theme ()
		{
		}

		public static Color LightFillGray = Color.FromArgb(206, 212, 223);

		//public static Color HourLabelColor = Color.FromArgb (85, 85, 85);
		public static Color HourLabelColor = SolarizedColors.Base02;

		public static IGUIFont HourFont { get; internal set; }
		public static IGUIFont MinuteFont { get; internal set; }
	}
}

