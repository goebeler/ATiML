package core;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Custom windowifier for the activity recognition dataset.
 * New attributes are mean and standard deviation of the sensor data,
 * a 'mean' of the WEKA-internal values for the devices and the class
 * with the highest frequency.
 * @author Florian Bethe
 *
 */
public class ActivityWindowifier implements Windowifier {

	private ArrayList<Attribute> attributes;
	private Instances structure;
	
	/**
	 * Constructor.
	 * Stores the wanted list of attributes as well as class attribute. This is to be
	 * customized for different data sets!
	 * @param classAttribute Attribute to be used as class attribute
	 */
	public ActivityWindowifier(Attribute classAttribute) {
		attributes = new ArrayList<Attribute>(8);
		attributes.add(new Attribute("xMean"));
		attributes.add(new Attribute("xStDev"));
		attributes.add(new Attribute("yMean"));
		attributes.add(new Attribute("yStDev"));
		attributes.add(new Attribute("zMean"));
		attributes.add(new Attribute("zStDev"));
		attributes.add(new Attribute("Device"));
		attributes.add(classAttribute);
		
		structure = new Instances("ActivityRecognition", attributes, 0);
		structure.setClass(classAttribute);
	}
	
	@Override
	public Instances getDataStructure() {
		return structure;
	}

	@Override
	public Instance windowify(List<Instance> instances) {
		// AttrValues accumulates the attribute values of x, y, z for the mean and std. deviation and device
		double[] attrValues = new double[attributes.size() - 1];
		// Accumulate the occurrence of each class
		// This is cumbersome since WEKA offers no direct way of counting the class frequency...
		HashMap<Double, Integer> classCount = new HashMap<Double, Integer>();
		
		for(Instance currInstance : instances) {
			// x, y, z mean
			attrValues[0] += currInstance.value(3);
			attrValues[2] += currInstance.value(4);
			attrValues[4] += currInstance.value(5);
			
			// x, y, z std. deviation
			attrValues[1] += currInstance.value(3)*currInstance.value(3);
			attrValues[3] += currInstance.value(4)*currInstance.value(4);
			attrValues[5] += currInstance.value(5)*currInstance.value(5);
			
			// Device 'mean'
			attrValues[6] += currInstance.value(8);
			
			// Update class count
			if(classCount.containsKey(currInstance.classValue())) {
				Integer currClassCount = classCount.get(currInstance.classValue());
				classCount.put(currInstance.classValue(), currClassCount + 1);
			} else {
				classCount.put(currInstance.classValue(), 1);
			}
		}

		// x, y, z mean
		attrValues[0] /= (double)(instances.size());
		attrValues[2] /= (double)(instances.size());
		attrValues[4] /= (double)(instances.size());
		
		// x, y, z std. deviation
		attrValues[1] = attrValues[1] / (double)(instances.size()) + attrValues[0]*attrValues[0];
		attrValues[3] = attrValues[3] / (double)(instances.size()) + attrValues[2]*attrValues[2];
		attrValues[5] = attrValues[5] / (double)(instances.size()) + attrValues[4]*attrValues[4];
		
		// Device 'mean'
		attrValues[6] /= (double)(instances.size());
		
		// Set up the new 'windowed' instance with the computed attributes
		Instance windowedInstance = new DenseInstance(attributes.size());
		windowedInstance.setDataset(structure);
		for(int i = 0; i < attrValues.length; i++)
			windowedInstance.setValue(attributes.get(i), attrValues[i]);
		
		// Select the class value with the highest frequency and name it winner (for the window)!
		double val = 0;
		int top = 0;
		for(Map.Entry<Double, Integer> pair : classCount.entrySet()) {
			if(pair.getValue() >= top) {
				top = pair.getValue();
				val = pair.getKey();
			}
		}
		
		windowedInstance.setClassValue(val);
		
		return windowedInstance;
	}

	@Override
	public Instances windowify(Instances instances, int windowSize, int windowOverlap) {
		Instances windows = new Instances(structure, instances.size() / (windowSize - windowOverlap));
		
		// Iterate over the possible (sequential) windows
		for(int i = 0; i < instances.size() - windowSize; i += windowSize - windowOverlap) {
			windows.add(this.windowify(new Instances(instances, i, windowSize)));
		}
		return windows;
	}

	
}
