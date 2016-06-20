package core;

import java.util.List;

import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;

/**
 * Generalized interface for the online and offline evaluators.
 * @author Florian Bethe
 *
 */
public interface Evaluator {
	/**
	 * Evaluates the classifiers after being trained on every training instance (in order).
	 * @param instanceStream List of instances to be used for training
	 * @return Evaluation object for each classifier
	 * @throws Exception
	 */
	public List<Evaluation> evaluateCumulated(Instances instanceStream) throws Exception;
	
	/**
	 * Evaluates the classifiers after each stepSize instances added to the training set.
	 * For offline classifiers, the classifier has to be retrained after each added instance batch.
	 * @param instanceStream List of instances for training
	 * @param stepSize Number of instances to be added to the training set before re-evaluation
	 * @return Evaluation object for each classifier for each evaluation
	 * @throws Exception
	 */
	public List<List<Evaluation>> evaluate(Instances instanceStream, int stepSize) throws Exception;
	
	/**
	 * Class name of an added classifier.
	 * @param index Index of the classifier
	 * @return Simple class name of the classifier
	 */
	public String getClassifierName(int index);
}
