package core;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.RemoveWithValues;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
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
			Log.enable();
			
			String fileName = args[0];
			System.out.println("Loading file '" + fileName + "'...");
			Test t = new Test(fileName);
			
			Log.log("Loaded attributes:");
			for(Attribute a : t.getAttributes())
				Log.log("\t"+a.toString());
			
			System.out.println("Filtering data...");
			t.removeAttributes(new String[]{"Index", "Arrival_Time", "Creation_Time",
					"User", "Model"});
			t.removeInstancesWithValue("gt", "null");

			t.setClassAttribute("gt");

			Log.log("Final attributes:");
			for(Attribute a : t.getAttributes())
				Log.log("\t"+a.toString());
			System.out.println("Class attribute: " + t.getClassAttribute());
			
			IBk c1 = new IBk();
			c1.setKNN(13);

			J48 c2 = new J48();
			c2.setUnpruned(true);
			
			RandomForest c3 = new RandomForest();
			c3.setNumFeatures(4);

			J48 c4 = new J48();
			c4.setUnpruned(false);
			c4.setReducedErrorPruning(true);
			
			IBk c5 = new IBk();
			c5.setKNN(3);
			
			t.addClassifier(c1);
			t.addClassifier(c2);
			t.addClassifier(c3);
			t.addClassifier(c4);
			t.addClassifier(c5);
			
			t.setTrainingShare(0.5);
			t.setTestShare(0.5);
			
			System.out.println("Testing classifiers...");
			ArrayList<Evaluation> evals = t.evaluateClassifiers();
			
			System.out.println("\nResults\n-------------------------\n\n");
			for(int i = 0; i < evals.size(); i++) {
				System.out.println("Classifier '" + t.getClassifier(i).getClass().getSimpleName() + ":");
				System.out.println("\tCorrectly classified: " + (1 - evals.get(i).errorRate()));
			}
			
		} catch (IOException e) {
			e.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public Test(String fileName) throws IOException {
		CSVLoader loader = new CSVLoader();
		loader.setSource(new File(fileName));
		data = loader.getDataSet();
		
		classifiers = new ArrayList<Classifier>();
		
		trainingPercent = 50.0;
		testPercent = 50.0;
	}
	
	public void setClassAttribute(String name) throws NullPointerException {
		data.setClass(data.attribute(name));
	}
	
	public void removeInstancesWithValue(String attr, String value) throws Exception {
		RemoveWithValues remVal = new RemoveWithValues();
		
		remVal.setAttributeIndex(Integer.toString(data.attribute(attr).index()));
		remVal.setNominalIndicesArr(new int[]{data.attribute(attr).indexOfValue(value)});
		remVal.setMatchMissingValues(true);
		remVal.setInputFormat(data);
		
		data = Filter.useFilter(data, remVal);
	}
	
	public void removeAttributes(String[] attrs) throws Exception{
		Remove rem = new Remove();
		
		int[] indices = new int[attrs.length];
		for(int i = 0; i < attrs.length; i++) {
			Attribute currAttr = data.attribute(attrs[i]);
			if(currAttr == null) {
				System.err.println("Error: attribute '" + attrs[i] + "' doesn't exist."
						+ " No attribute will be removed!");
				return ;
			}
			indices[i] = currAttr.index();
		}
		rem.setAttributeIndicesArray(indices);
		rem.setInputFormat(data);
		
		data = Filter.useFilter(data, rem);
	}
	
	public List<Attribute> getAttributes() {
		return Collections.list(data.enumerateAttributes());
	}
	
	public Attribute getClassAttribute() {
		return data.classAttribute();
	}
	
	public void addClassifier(Classifier classifier) {
		classifiers.add(classifier);
	}
	
	public Classifier getClassifier(int index) {
		return classifiers.get(index);
	}
	
	public void setTrainingShare(double percent) {
		trainingPercent = percent;
	}
	
	public void setTestShare(double percent) {
		testPercent = percent;
	}
	
	public ArrayList<Evaluation> evaluateClassifiers() throws Exception {
		ArrayList<Evaluation> evals = new ArrayList<Evaluation>();
		
		if(trainingPercent + testPercent > 100.0)
			testPercent = 100.0 - trainingPercent;
		int trainingSize = (int)(data.numInstances() * trainingPercent / 100.0);
		int testSize = (int)(data.numInstances() * testPercent / 100.0);
		
		data.randomize(new java.util.Random(System.currentTimeMillis()));
		
		Instances trainingSet = new Instances(data, 0, trainingSize);
		Instances testSet = new Instances(data, trainingSize, testSize);

		Log.log("Training instances: " + trainingSize);
		Log.log("Test instances: " + testSize);
		
		for(Classifier c : classifiers) {
			Log.log("\tTraining classifier '" + c.getClass().getSimpleName() + "'...");
			c.buildClassifier(trainingSet);
			
			Log.log("\tEvaluating classifier '" + c.getClass().getSimpleName() + "'...");
			Evaluation currEval = new Evaluation(trainingSet);
			currEval.evaluateModel(c, testSet);
			
			evals.add(currEval);
		}
		
		return evals;
	}
}