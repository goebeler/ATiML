package util;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;

public class Log {
	private static boolean consoleEnabled = false;
	private static String protocol = "";
	
	public static void enableConsole() {
		consoleEnabled = true;
	}
	
	public static void disableConsole() {
		consoleEnabled = false;
	}
	
	public static void log(String s) {
		s = "[" + new SimpleDateFormat("EEE, dd MMM YYYY HH:mm:ss z").format(new java.util.Date()) + "] - " + s;
		protocol += s;
		if(consoleEnabled)
			System.out.println(s);
	}
	
	public static String getProtocol() {
		return protocol;
	}
	
	public static void saveProtocol(String fileName) throws FileNotFoundException {
		PrintWriter writer = new PrintWriter(fileName);
		writer.println(protocol);
		writer.close();
	}
}
