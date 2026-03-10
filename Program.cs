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
        [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern bool SetDllDirectory(string path);

        public static string DataDir { get; private set; }
        public static string PdbDir { get; private set; }

        //public static MainForm MainWindow { get; private set; }
        public static ApplicationWindow MainWindow { get; private set; }

        [STAThread]
        static void Main(string[] args)
        {
            AppDomain.CurrentDomain.UnhandledException += new UnhandledExceptionEventHandler(ExceptionUnhandled);
            AppDomain.CurrentDomain.FirstChanceException += AppDomain_CurrentDomain_FirstChanceException;

            Logging.SetupLogging(LogLevels.Verbose, LogTargets.Console);

            DataDir = "/home/detlef/ProteinFaltung";
            PdbDir = "/home/detlef/ProteineDataBase";

            int pf = (int)Environment.OSVersion.Platform;
            if (pf != 4 && pf != 6 && pf != 128)
            {
                string path = Path.GetDirectoryName(Assembly.GetEntryAssembly().Location);
                path = Path.Combine(path, IntPtr.Size == 8 ? "x64" : "x86");
                if (!SetDllDirectory(path))
                    throw new System.ComponentModel.Win32Exception();
            }

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
