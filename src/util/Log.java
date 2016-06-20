package util;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;

/**
 * Simple logger class documenting the loading and evaluation process.
 * @author windowOverlap
 */
public class Log {
	private static boolean consoleEnabled = false;
	private static String protocol = "";
	
	/**
	 * Enables logger output to the standard std::out console.
	 */
	public static void enableConsole() {
		consoleEnabled = true;
	}

	/**
	 * Disables logger output to the standard std::out console.
	 */
	public static void disableConsole() {
		consoleEnabled = false;
	}
	
	/**
	 * Logs a string.
	 * @param s String to be logged
	 */
	public static void log(String s) {
		s = "[" + new SimpleDateFormat("EEE, dd MMM YYYY HH:mm:ss z").format(new java.util.Date()) + "] - " + s;
		protocol += s + '\n';
		if(consoleEnabled)
			System.out.println(s);
	}
	
	/**
	 * Returns the accumulated log.
	 * @return Full log
	 */
	public static String getProtocol() {
		return protocol;
	}
	
	/**
	 * Saves the accumulated protocol to a given file.
	 * @param fileName Name of file
	 * @throws FileNotFoundException
	 */
	public static void saveProtocol(String fileName) throws FileNotFoundException {
		PrintWriter writer = new PrintWriter(fileName);
		writer.println(protocol);
		writer.close();
	}
}
