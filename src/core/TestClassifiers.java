package core;

import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemoveWithValues;
import weka.classifiers.Classifier;
import weka.classifiers.UpdateableClassifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.bayes.NaiveBayesMultinomialUpdateable;
import weka.classifiers.bayes.NaiveBayesUpdateable;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import util.Log;

/**
 * Main class for the classifier evaluation.
 * Loads the data and the specified classifiers and evaluates them on a 50/50 split after
 * applying sliding window to the data set.
 * @author Florian Bethe
 *
 */
public class TestClassifiers {
	
	public static void main(String[] args) {
		if(args.length < 1) {
			System.err.println("Usage: java " + TestClassifiers.class.getName() + " <filename> [-online] [-classifier name params...]*");
			System.exit(0);
		}
		
		try {
			// Enable the logger to output to the standard console
			Log.enableConsole();
			
			// Load the data file given via command line
			String fileName = args[0];

			// Load the CSV file containing the data
			Log.log("Loading data from file '" + fileName + "'...");
			CSVLoader loader = new CSVLoader();
			loader.setSource(new File(fileName));
			Instances data = loader.getDataSet();
			
			// Filter out the 'null' values from the 'gt' class attribute
			Log.log("Filtering out missing values...");
			RemoveWithValues rem = new RemoveWithValues();
			// + 1 since WEKA indices start at 1 when given as string
			rem.setAttributeIndex(Integer.toString(data.attribute("gt").index() + 1));
			rem.setNominalIndicesArr(new int[]{data.attribute("gt").indexOfValue("null")});
			rem.setInputFormat(data);
			data = Filter.useFilter(data, rem);
			
			data.setClass(data.attribute("gt"));
			
			// Apply sliding window to make use of the time component of the sequential data
			ActivityWindowifier windowifier = new ActivityWindowifier(data.classAttribute());
			data = windowifier.windowify(data, 256, 128);
			
			// Randomize the order of the windows
			data.randomize(new Random((long)(Math.random() * System.currentTimeMillis())));

			// Split up the data into training and testing (50 / 50)
			Instances trainingData = new Instances(data, 0, data.numInstances() / 2);
			Instances testData = new Instances(data, data.numInstances() / 2, data.numInstances() / 2);
			
			// Parse the command line arguments:
			// [-online] [-classifier name params...]*
			String parameters = String.join(" ", args);
			boolean online = parameters.contains("-online");
			List<String> classifierParams = new ArrayList<String>(Arrays.asList(parameters.split("-classifier ")));
			classifierParams = classifierParams.subList(1, classifierParams.size());
			
			// Set up the correct evaluator
			Evaluator eval = null;
			if(online) {
				eval = new OnlineEvaluation(testData);
				for(UpdateableClassifier classifier : TestClassifiers.parseUpdateableClassifiersFromCommandline(classifierParams)) {
					((OnlineEvaluation)eval).addClassifier(classifier);
				}
			} else {
				eval = new OfflineEvaluation(testData);
				for(Classifier classifier : TestClassifiers.parseClassifiersFromCommandline(classifierParams)) {
					((OfflineEvaluation)eval).addClassifier(classifier);
				}
			}
			
			int index = 0;
			for(Evaluation e : eval.evaluateCumulated(trainingData)) {
				Log.log(eval.getClassifierName(index++) + ":\n" + e.toSummaryString() + "\n" + e.confusionMatrix());
			}
			
			Log.saveProtocol("protocols/Log" + System.currentTimeMillis() + ".txt");
			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * Parse the wanted classifiers out of the specified cmd parameters.
	 * 
	 * @param classifierParams List of command line parameters inbetween '-classifier '.
	 * @return List of 'normal' classifiers
	 */
	private static ArrayList<Classifier> parseClassifiersFromCommandline(List<String> classifierParams) {
		ArrayList<Classifier> classifiers = new ArrayList<Classifier>();
		
		for(String option : classifierParams) {
			System.out.println(option);
			option = option.toLowerCase();
			String[] values = option.split(" ");
			
			switch(values[0]) {
			case "knn":
				int k = 1;
				if(values.length >= 2)
					k = Integer.parseInt(values[1]);
				classifiers.add(new IBk(k));
				break;
			case "randomforest":
				RandomForest rf = new RandomForest();
				if(values.length >= 2)
					rf.setMaxDepth(Integer.parseInt(values[1]));
				if(values.length >= 3)
					rf.setNumFeatures(Integer.parseInt(values[2]));
				classifiers.add(rf);
				break;
			case "naivebayes":
				NaiveBayes nb = new NaiveBayes();
				if(values.length >= 2)
					nb.setUseKernelEstimator(Boolean.parseBoolean(values[1]));
				classifiers.add(nb);
				break;
			case "naivebayesmulti":
				classifiers.add(new NaiveBayesMultinomial());
				break;
			case "j48":
				J48 tree = new J48();
				if(values.length >= 2)
					tree.setUnpruned(Boolean.parseBoolean(values[1]));
				classifiers.add(tree);
				break;
			}
		}
		
		return classifiers;
	}
	
	/**
	 * Parse the wanted updatable (online) classifiers out of the specified cmd parameters.
	 * @param classifierParams List of command line parameters inbetween '-classifier '.
	 * @return List of online classifiers
	 */
	private static ArrayList<UpdateableClassifier> parseUpdateableClassifiersFromCommandline(List<String> classifierParams) {
		ArrayList<UpdateableClassifier> classifiers = new ArrayList<UpdateableClassifier>();
		
		System.out.println(classifierParams);
		
		for(String option : classifierParams) {
			System.out.println(option);
			option = option.toLowerCase();
			String[] values = option.split(" ");
			
			switch(values[0]) {
			case "knn":
				int k = 1;
				if(values.length >= 2)
					k = Integer.parseInt(values[1]);
				classifiers.add(new IBk(k));
				break;
			case "naivebayes":
				NaiveBayesUpdateable nb = new NaiveBayesUpdateable();
				if(values.length >= 2)
					nb.setUseKernelEstimator(Boolean.parseBoolean(values[1]));
				classifiers.add(nb);
				break;
			case "naivebayesmulti":
				classifiers.add(new NaiveBayesMultinomialUpdateable());
				break;
			}
		}
		
		return classifiers;
	}
}