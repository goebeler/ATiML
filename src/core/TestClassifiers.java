package core;

import weka.core.Attribute;
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
			rem.setModifyHeader(true);
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
			String clsNames = "";
			if(online) {
				eval = new OnlineEvaluation(testData);
				for(UpdateableClassifier classifier : TestClassifiers.parseUpdateableClassifiersFromCommandline(classifierParams)) {
					((OnlineEvaluation)eval).addClassifier(classifier);
					clsNames += classifier.getClass().getSimpleName() + " ";
				}
			} else {
				eval = new OfflineEvaluation(testData);
				for(Classifier classifier : TestClassifiers.parseClassifiersFromCommandline(classifierParams)) {
					((OfflineEvaluation)eval).addClassifier(classifier);
					clsNames += classifier.getClass().getSimpleName() + " ";
				}
			}
			
			Log.log("Selected classifiers: " + clsNames + "...");
			
			if(parameters.contains("-steps")) {
				int stepSize = Integer.parseInt(parameters.split("-steps ")[1].split(" ")[0]);
				
				Log.log("Evaluation step size: " + stepSize + "...");
				
				int currentStep = stepSize;
				
				for(List<Evaluation> evals : eval.evaluate(trainingData, stepSize)) {
					Log.log("Current training set size: " + currentStep + "\n------------------------\n");
					int index = 0;
					for(Evaluation e : evals) {
						Log.log(eval.getClassifierName(index++) + ":\n" + e.toSummaryString() + "\n"
									+ printConfusionMatrix(trainingData.classAttribute(), e.confusionMatrix()) + "\n");
					}
					currentStep += stepSize;
				}
			} else {
				int index = 0;
				for(Evaluation e : eval.evaluateCumulated(trainingData)) {
					Log.log(eval.getClassifierName(index++) + ":\n" + e.toSummaryString() + "\n"
								+ printConfusionMatrix(trainingData.classAttribute(), e.confusionMatrix()));
				}
			}
			
			Log.saveProtocol("protocols/Log" + System.currentTimeMillis() + ".txt");
			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * Prints the given confusion matrix with respect to the class attribute to a string.
	 * Each value holding cell is three tabs wide.
	 * @param classAttr Class attribute for the confusion matrix
	 * @param matrix Confusion matrix to be printed
	 * @return String showing the confusion matrix
	 */
	private static String printConfusionMatrix(Attribute classAttr, double[][] matrix) {
		final int cellSize = 16;
		
		double[] totalY = new double[matrix.length];
		String[] cellSeparator = new String[classAttr.numValues()];
		
		// Top row with the attribute names
		String res = "\t\t\t";
		for(int x = 0; x < matrix.length; x++) {
			cellSeparator[x] = new String(new char[(cellSize - classAttr.value(x).length() - 1)/8]).replace("\0", "\t");
			res += classAttr.value(x) + cellSeparator[x];
			for(int y = 0; y < matrix.length; y++) {
				totalY[y] += matrix[y][x];
			}
		}
		
		// Matrix
		res += "\r\n";
		for(int y = 0; y < matrix.length; y++) {
			// Class names at the beginning or row
			res += "\t" + classAttr.value(y) + cellSeparator[y];
			
			double totalX = 0;
			for(int x = 0; x < matrix[y].length; x++) {
				res += (int)(matrix[y][x]) + "\t\t";
				totalX += matrix[y][x];
			}
			
			// Row sum at end of row
			res += (int)(totalX) + "\r\n";
		}
		
		// Bottom row with column sum
		res += "\t\t\t";
		double total = 0;
		for(int i = 0; i < totalY.length; i++) {
			res += (int)(totalY[i]) + "\t\t";
			total += totalY[i];
		}
		
		// Overall total
		res += (int)(total);
		
		return res;
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
		
		for(String option : classifierParams) {
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