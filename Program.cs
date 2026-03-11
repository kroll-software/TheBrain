using System;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Diagnostics;
using KS.Foundation;
using SummerGUI;

namespace TheBrain
{
    public class MainClass
    {        
        //public static MainForm MainWindow { get; private set; }
        public static ApplicationWindow MainWindow { get; private set; }

        [STAThread]
        static void Main(string[] args)
        {
            AppDomain.CurrentDomain.UnhandledException += new UnhandledExceptionEventHandler(ExceptionUnhandled);
            AppDomain.CurrentDomain.FirstChanceException += AppDomain_CurrentDomain_FirstChanceException;

            Logging.SetupLogging(LogLevels.Verbose, LogTargets.Console);            

            using (MainForm wnd = new MainForm())
            {
                MainWindow = wnd;
                // limit rate to a value between 30 and 60 Hz
                //float rate = Math.Min(60, Math.Max(30, OpenTK.DisplayDevice.Default.RefreshRate));
                float rate = 30;
                wnd.Run(rate, rate);
            }
        }

        static void ExceptionUnhandled(object sender, UnhandledExceptionEventArgs args)
        {
            Exception e = (Exception)args.ExceptionObject;
            e.LogError("UNHANDLED Exception");
        }

        static void AppDomain_CurrentDomain_FirstChanceException(object sender, System.Runtime.ExceptionServices.FirstChanceExceptionEventArgs e)
        {
            e.Exception.LogError("First-Chance Exception");
        }
    }
}
