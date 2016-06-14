package util;

public class Log {
	private static boolean isEnabled = false;
	
	public static void enable() {
		isEnabled = true;
	}
	
	public static void disable() {
		isEnabled = false;
	}
	
	public static void log(String s) {
		if(isEnabled)
			System.out.println(s);
	}
}
