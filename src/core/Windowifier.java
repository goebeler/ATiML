package core;

import java.util.List;

import weka.core.Instance;
import weka.core.Instances;

/**
 * Interface for applying sliding window to data sets.
 * In sliding window a window of a given size of slided over the instances in order
 * to transform sequential data (for example with a time component) into non-sequential data.
 * This is needed to allow standard supervised classifiers to classify the data.
 * @author Florian Bethe
 *
 */
public interface Windowifier {
	/**
	 * Returns the data structure of the instances of the window sliding.
	 * @return Empty set of instances with class and normal attributes
	 */
	public Instances getDataStructure();
	
	/**
	 * Applies sliding window for a single window.
	 * The returned instance has to be of the same structure as returned by getDataStructure().
	 * @param instances List of instances part of the window
	 * @return Single instance representing the window
	 */
	public Instance windowify(List<Instance> instances);
	
	/**
	 * Applies sliding window to a full data set.
	 * The overlap determines how many instances two adjacent windows share.
	 * @param instances Full data set
	 * @param windowSize Size of the window
	 * @param windowOverlap Overlap between two adjacent windows
	 * @return New data set comprised of windows after transformation
	 */
	public Instances windowify(Instances instances, int windowSize, int windowOverlap);
}
