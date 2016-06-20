package core;

import java.util.ArrayList;
import java.util.List;

import util.Log;
import weka.classifiers.Classifier;
import weka.classifiers.UpdateableClassifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Evaluator for online classifiers.
 * @author Florian Bethe
 *
 */
public class OnlineEvaluation implements Evaluator {
	private List<UpdateableClassifier> classifiers;
	
	private Instances testSet;
	
	public OnlineEvaluation(Instances testSet) {
		classifiers = new ArrayList<UpdateableClassifier>();
		
		this.testSet = testSet;
	}

	@Override
	public List<Evaluation> evaluateCumulated(Instances instanceStream) throws Exception {
		// Accumulate the training set for the evaluation (needs it for the statistics...)
		Instances trainingData = new Instances(testSet, 0, 0);
		
		// Set up the classifiers internal data structures
		for(UpdateableClassifier classifier : classifiers)
			((Classifier)classifier).buildClassifier(trainingData);
		
		// Train the classifiers on the data
		for(Instance instance : instanceStream) {
			for(UpdateableClassifier classifier : classifiers) {
				classifier.updateClassifier(instance);
			}
			
			trainingData.add(instance);
		}
		
		// Evaluate them on the test data
		ArrayList<Evaluation> evals = new ArrayList<Evaluation>();
		for(UpdateableClassifier classifier : classifiers) {
			Log.log("Evaluating " + classifier.getClass().getSimpleName() + "...");
			Evaluation currEval = new Evaluation(trainingData);
			currEval.evaluateModel((Classifier)(classifier), testSet);
			evals.add(currEval);
		}
		
		return evals;
	}

	@Override
	public List<List<Evaluation>> evaluate(Instances instanceStream, int stepSize) throws Exception {
		stepSize = Math.max(1, Math.min(stepSize, instanceStream.size()));
		
		// Accumulate the training set for the evaluation (needs it for the statistics...)
		Instances trainingData = new Instances(testSet, 0, 0);

		// Set up the classifiers internal data structures
		for(UpdateableClassifier classifier : classifiers)
			((Classifier)classifier).buildClassifier(trainingData);
		
		List<List<Evaluation>> evals = new ArrayList<List<Evaluation>>();
		
		int currTrainingSize = 0;
		
		for(Instance instance : instanceStream) {
			// Update the classifiers
			for(UpdateableClassifier classifier : classifiers) {
				classifier.updateClassifier(instance);
			}
			
			trainingData.add(instance);

			// Evaluate them on the test data after each stepSize training set size increment
			if(++currTrainingSize % stepSize == 0) {
				ArrayList<Evaluation> currEvals = new ArrayList<Evaluation>(classifiers.size());
				
				for(UpdateableClassifier classifier : classifiers) {
					Evaluation currEval = new Evaluation(trainingData);
					currEval.evaluateModel((Classifier)(classifier), testSet);
					currEvals.add(currEval);
				}
				
				evals.add(currEvals);
			}
		}
		
		return evals;
	}

	@Override
	public String getClassifierName(int index) {
		return classifiers.get(index).getClass().getSimpleName();
	}
	
	public void addClassifier(UpdateableClassifier classifier) {
		classifiers.add(classifier);
	}
}
