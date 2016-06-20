package core;

import java.util.ArrayList;
import java.util.List;

import util.Log;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;

/**
 * Evaluator for offline classifiers.
 * @author Florian Bethe
 *
 */
public class OfflineEvaluation implements Evaluator {
	private List<Classifier> classifiers;
	
	private Instances testSet;
	
	public OfflineEvaluation(Instances testSet) {
		classifiers = new ArrayList<Classifier>();
		this.testSet = testSet;
	}

	@Override
	public List<Evaluation> evaluateCumulated(Instances trainingData) throws Exception {
		// Train the classifiers on the full data
		for(Classifier classifier : classifiers)
			classifier.buildClassifier(trainingData);
		
		// Evaluate the classifiers on test data
		List<Evaluation> evals = new ArrayList<Evaluation>();
		for(Classifier classifier : classifiers) {
			Log.log("Evaluating " + classifier.getClass().getSimpleName() + "...");
			
			Evaluation currEval = new Evaluation(trainingData);
			currEval.evaluateModel(classifier, testSet);
			
			evals.add(currEval);
		}
		
		return evals;
	}

	@Override
	public List<List<Evaluation>> evaluate(Instances instanceStream, int stepSize) throws Exception {
		stepSize = Math.max(1, Math.min(stepSize, instanceStream.size()));
		
		List<List<Evaluation>> evals = new ArrayList<List<Evaluation>>();
		
		for(int i = stepSize; i <= instanceStream.numInstances(); i++) {
			Instances trainingData = new Instances(instanceStream, 0, i);
			
			// Training
			for(Classifier classifier : classifiers) {
				classifier.buildClassifier(trainingData);
			}

			// Evaluation
			List<Evaluation> currEvals = new ArrayList<Evaluation>(classifiers.size());
			for(Classifier classifier : classifiers) {
				Log.log("Evaluating " + classifier.getClass().getSimpleName() + "...");
				
				Evaluation currEval = new Evaluation(trainingData);
				currEval.evaluateModel(classifier, testSet);
				
				currEvals.add(currEval);
			}
			
			evals.add(currEvals);
		}
		
		return evals;
	}

	@Override
	public String getClassifierName(int index) {
		return classifiers.get(index).getClass().getSimpleName();
	}
	
	public void addClassifier(Classifier classifier) {
		classifiers.add(classifier);
	}
}
