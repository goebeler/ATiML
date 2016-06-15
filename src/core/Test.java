package core;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.CSVSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.RemoveWithValues;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.PrincipalComponents;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IBk;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.trees.RandomForest;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import util.Log;

public class Test {
	Instances data;
	ArrayList<Classifier> classifiers;
	double trainingPercent;
	double testPercent;
	
	public static void main(String[] args) {
		if(args.length < 1) {
			System.out.println("Usage: java " + Test.class.getName() + " <filename>");
			System.exit(0);
		}
		
		try {
			// Enable the logger to output to the standard console
			Log.enableConsole();
			
			// Load the data file given via command line
			String fileName = args[0];
			Test t = new Test(fileName, "");

			// Remove the 'null' value, as we have no effective way of utilizing those instances
			t.removeInstancesWithValue("gt", "null");
			t.setClassAttribute("gt");
	
			t.applySlidingWindow(256, 128);
			//t.saveInstances("data/Phones_gyroscope_window256-128.csv");
			
			
			// K nearest neighbor classifier with k = 9
			IBk c1 = new IBk(9);
			RandomForest c2 = new RandomForest();
			
			// Naive Bayes with kernel density estimation
			NaiveBayes c3 = new NaiveBayes();
			c3.setUseKernelEstimator(true);
			
			// A (guessed) neural network with hidden units
			MultilayerPerceptron c4 = new MultilayerPerceptron();
			c4.setAutoBuild(true);
			c4.setHiddenLayers("4,6");
			
			t.addClassifier(c1);
			t.addClassifier(c2);
			t.addClassifier(c3);
			t.addClassifier(c4);
			
			t.setTrainingShare(66);
			t.setTestShare(34);
			t.randomizeData();

			//t.applyPCA(6);
			Log.log(t.summarizeEvaluation(t.evaluateClassifiers()));
			
			Log.saveProtocol("protocols/Log" + System.currentTimeMillis() + ".txt");
			
		} catch (IOException e) {
			e.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * Constructor.
	 * Loads the data from the specified file. If needed, a missing value can be specified
	 * which may receive special treatment.
	 * @param fileName Data file name
	 * @param missingValue Missing value
	 * @throws IOException
	 */
	public Test(String fileName, String missingValue) throws IOException {
		Log.log("Loading data from file '" + fileName + "'...");
		
		CSVLoader loader = new CSVLoader();
		if(!missingValue.isEmpty())
			loader.setMissingValue(missingValue);
		loader.setSource(new File(fileName));
		data = loader.getDataSet();
		
		String attributeList = "Loaded data with " + data.numInstances() + " instances and " + data.numAttributes() + " attributes:";
		for(Attribute a : Collections.list(data.enumerateAttributes()))
			attributeList += "\n\t"+a.toString();
		Log.log(attributeList);
		
		classifiers = new ArrayList<Classifier>();
		
		trainingPercent = 50.0;
		testPercent = 50.0;
	}
	
	/**
	 * Applies a sliding window algorithm to our specific data set.
	 * The data is assumed to be ordered as a (time) sequence. Thus, it will be merged into
	 * windows to capture this property. The attributes of the windows are mean and standard
	 * deviation of the x,y,z coordinates of the sensor. The corresponding class feature will
	 * be decided by majority vote within the window.
	 * @param windowSize Size of the windows
	 * @param overlap Overlap between individual windows
	 * @throws Exception
	 */
	public void applySlidingWindow(int windowSize, int overlap) throws Exception {
		Log.log("Applying sliding window with size=" + windowSize + " and overlap=" + overlap + "...");
		
		// Define the new attributes: mean and standard deviation of x, y, z in each window
		// TODO: Bring the device into it?
		ArrayList<Attribute> attrs = new ArrayList<Attribute>(8);
		attrs.add(new Attribute("xMean"));
		attrs.add(new Attribute("xStd"));
		attrs.add(new Attribute("yMean"));
		attrs.add(new Attribute("yStd"));
		attrs.add(new Attribute("zMean"));
		attrs.add(new Attribute("zStd"));
		attrs.add(new Attribute("Device"));
		attrs.add(data.classAttribute());
		
		// Prepare the new instance set by reserving enough space for every window
		Instances newInstances = new Instances(data.relationName(), attrs, (int) Math.ceil(data.numInstances() / (double)(overlap)));
		newInstances.setClass(data.classAttribute());
		
		// Iterate over every window consecutively
		for(int i = 0; i < data.numInstances(); i += windowSize-overlap) {
			// TODO: should we leave the (possibly much smaller) window at the end of the sequence out?
			//if((i+256) >= data.numInstances())
			//	break;

			// Accumulate the x,y,z values to compute mean and st. deviation
			double[] attr = {0, 0, 0, 0, 0, 0, 0};
			// Prevent the window for extending beyond the size of the instance set
			int currSize = Math.min(windowSize, data.numInstances()-i);
			
			Instances windowInstances = new Instances(data, i, currSize);
			for(Instance inst : Collections.list(windowInstances.enumerateInstances())) {
				attr[0] += inst.value(3);
				attr[1] += inst.value(3)*inst.value(3);
				attr[2] += inst.value(4);
				attr[3] += inst.value(4)*inst.value(4);
				attr[4] += inst.value(5);
				attr[5] += inst.value(5)*inst.value(5);
				attr[6] += inst.value(8);
			}

			// Compute the means and st. deviations
			attr[0] /= (double)(currSize);
			attr[2] /= (double)(currSize);
			attr[4] /= (double)(currSize);
			attr[1] = attr[1] / (double)(currSize-1) - attr[0]*attr[0];
			attr[3] = attr[3] / (double)(currSize-1) - attr[2]*attr[2];
			attr[5] = attr[5] / (double)(currSize-1) - attr[4]*attr[4];
			attr[6] /= (double)(currSize);
			
			// Build the new instance with the custom attributes
			Instance newInstance = new DenseInstance(attrs.size());
			newInstance.setDataset(newInstances);
			for(int j = 0; j < attrs.size() - 1; j++)
				newInstance.setValue(attrs.get(j), attr[j]);

			// Get the class label of the window by majority rule
			DecisionTable majorityClassifier = new DecisionTable();
			majorityClassifier.buildClassifier(windowInstances);
			// Since majority rules, it does not matter which instance we take
			newInstance.setClassValue(majorityClassifier.classifyInstance(data.instance(i+currSize/2)));
			
			newInstances.add(newInstance);
		}
		
		data = newInstances;
	}
	
	/**
	 * Sets the class attribute for the data.
	 * @param name Name of new class attribute
	 * @throws NullPointerException
	 */
	public void setClassAttribute(String name) throws NullPointerException {
		data.setClass(data.attribute(name));
		Log.log("Set class attribute to " + name);
	}
	
	/**
	 * Applies Principal Component Analysis to reduce the data's dimensionality.
	 * The new dimensions will be ranked by variance.
	 * @param numAttrs Number of new dimensions.
	 * @throws Exception
	 */
	public void applyPCA(int numAttrs) throws Exception {
		Log.log("Selecting " + numAttrs +" attributes via principal component analysis...");
		
		PrincipalComponents pca = new PrincipalComponents();
		pca.buildEvaluator(data);

		// The ranker orders the new dimensions by variance and outputs the top x
		Ranker ranker = new Ranker();
		if(numAttrs > 0)
			ranker.setNumToSelect(numAttrs);
		
		// Combine the PCA and the ranker for selection
		AttributeSelection selector = new AttributeSelection();
		selector.setEvaluator(pca);
		selector.setSearch(ranker);
		selector.SelectAttributes(data);
		
		data = selector.reduceDimensionality(data);
	}
	
	/**
	 * Remove instances with a certain value at an attribute.
	 * This method also removes values marked as missingValues.
	 * The data header will be modified to eleminate the removed values.
	 * @param attr Attribute name
	 * @param value Value to be removed
	 * @throws Exception
	 */
	public void removeInstancesWithValue(String attr, String value) throws Exception {
		Log.log("Removing all instances with value '" + value + "' in attribute '" + attr + "'...");
		
		RemoveWithValues remVal = new RemoveWithValues();
		
		// Find the attributes and the values internal indices for removal
		remVal.setAttributeIndex(Integer.toString(data.attribute(attr).index()+1));	// + 1 because WEKAs indices start at 1 when given in String form
		remVal.setNominalIndicesArr(new int[]{data.attribute(attr).indexOfValue(value)});
		remVal.setModifyHeader(true);
		remVal.setMatchMissingValues(true);
		
		remVal.setInputFormat(data);
		data = Filter.useFilter(data, remVal);
	}
	
	/**
	 * Removes the specified attributes from the data set.
	 * Misspelled or non-existing attributes will be ignored.
	 * @param attrs Array of attribute names to be removed.
	 * @throws Exception
	 */
	public void removeAttributes(String[] attrs) throws Exception{
		String logMsg = "Removing attributes {";
		for(int i = 0; i < attrs.length; i++)
			logMsg += "'" + attrs[i] + "', ";
		Log.log(logMsg.substring(0, logMsg.length() - 2) + "}...");
		
		// WEKA filter to remove attributes
		Remove rem = new Remove();
		
		// Obtain the indices used by WEKA for each specified attributes
		ArrayList<Integer> indices = new ArrayList<Integer>();
		for(int i = 0; i < attrs.length; i++) {
			Attribute currAttr = data.attribute(attrs[i]);
			
			// In case of e.g. misspellings abort the removal
			if(currAttr == null) {
				Log.log("Warning: attribute '" + attrs[i] + "' doesn't exist.");
				System.err.println("Warning: attribute '" + attrs[i] + "' doesn't exist.");
			} else {
				indices.add(currAttr.index());
			}
		}

		// Convert the arraylist to the needed array
		int[] indexArray = indices.stream().mapToInt(i -> i).toArray();
		
		// Set the index array and the data format
		rem.setAttributeIndicesArray(indexArray);
		rem.setInputFormat(data);
		
		
		data = Filter.useFilter(data, rem);
	}
	
	/**
	 * Returns the attributes of the data set.
	 * This does not include the class attribute, if one is present.
	 * @return List of regular attributes
	 */
	public List<Attribute> getAttributes() {
		return Collections.list(data.enumerateAttributes());
	}
	
	/**
	 * Returns the currently selected class attribute of the data set.
	 * @return Class attribute
	 */
	public Attribute getClassAttribute() {
		return data.classAttribute();
	}
	
	/**
	 * Adds a classifier to the list of classifiers.
	 * @param classifier New classifier
	 */
	public void addClassifier(Classifier classifier) {
		Log.log("Added classifier " + classifier.getClass().getSimpleName() + " to set.");
		classifiers.add(classifier);
	}
	
	/**
	 * Returns the classifiers stored.
	 * @return List of classifiers
	 */
	public ArrayList<Classifier> getClassifiers() {
		return classifiers;
	}

	/**
	 * Sets the share of training instances of the whole data set.
	 * @param percent Share of training set in percent
	 */
	public void setTrainingShare(double percent) {
		Log.log("Set training set share to " + percent + "%.");
		trainingPercent = percent;
	}
	
	/**
	 * Sets the share of test instances of the whole data set.
	 * @param percent Share of test set in percent
	 */
	public void setTestShare(double percent) {
		Log.log("Set test set share to " + percent + "%.");
		testPercent = percent;
	}
	
	/**
	 * Randomizes the order of instances stored in 'data'.
	 */
	public void randomizeData() {
		Log.log("Randomizing instance order...");
		// Since we only randomize once we use a temporary Random object
		data.randomize(new java.util.Random((long)(Math.random() * 10000000000000.0)));
	}
	
	/**
	 * Evaluates the classifiers stored in 'classifiers'.
	 * Each classifier will be trained on a percentage of the available data as specified by
	 * 'trainingPercent' and 'testPercent'. Note: these do not have to equal 100% but may not
	 * be larger than that.
	 * @return List of evaluation results, ordered by the order of classifiers tested.
	 * @throws Exception
	 */
	public ArrayList<Evaluation> evaluateClassifiers() throws Exception {
		ArrayList<Evaluation> evals = new ArrayList<Evaluation>();
		
		if(trainingPercent + testPercent > 100.0)
			testPercent = 100.0 - trainingPercent;
		
		// Compute the number of instances for both sets
		int trainingSize = (int)(data.numInstances() * trainingPercent / 100.0);
		int testSize = (int)(data.numInstances() * testPercent / 100.0);
		
		Instances trainingSet = new Instances(data, 0, trainingSize);
		Instances testSet = new Instances(data, trainingSize, testSize);

		Log.log("Classifier evaluation (" + trainingSize + " training | " + testSize + " testing)");
		
		int classifierCounter = 1;
		
		// Each classifier in the set gets trained and tested
		for(Classifier c : classifiers) {
			Log.log("\t(" + classifierCounter + " of " + classifiers.size() + ") Training classifier '" + c.getClass().getSimpleName() + "'...");
			c.buildClassifier(trainingSet);
			
			Log.log("\t(" + classifierCounter + " of " + classifiers.size() + ") Evaluating classifier '" + c.getClass().getSimpleName() + "'...");
			Evaluation currEval = new Evaluation(trainingSet);
			currEval.evaluateModel(c, testSet);
			
			evals.add(currEval);
			
			classifierCounter++;
		}
		
		return evals;
	}
	
	/**
	 * Saves a summary of the provided evaluation results for each classifier.
	 * @param evals Previously obtained evaulation results.
	 * @return result summary
	 */
	public String summarizeEvaluation(ArrayList<Evaluation> evals) {
		String resultString = "\n------------------------------------------------\n"
							+ "-------------------- RESULTS -------------------\n"
							+ "------------------------------------------------\n";

		for(int i = 0; i < evals.size(); i++) {
		String classifierName = "\nClassifier " + classifiers.get(i).getClass().getSimpleName() + '\n';
		resultString += classifierName
				+ new String(new char[classifierName.length() + 2]).replace('\0', '-') + '\n'
				+ evals.get(i).toSummaryString();
		}
		
		return resultString;
	}
	
	/**
	 * Saves the currently used set of instances to a CSV file.
	 * @param fileName Name of output file
	 * @throws IOException
	 */
	public void saveInstances(String fileName) throws IOException {
		Log.log("Saving " + data.numInstances() + " instances to file '" + fileName + "'...");
		CSVSaver writer = new CSVSaver();
		writer.setDestination(new File(fileName));
		writer.setFile(new File(fileName));
		writer.setInstances(data);
		writer.writeBatch();
	}
}