import java.util.*;

/**
 * Fill in the implementation details of the class DecisionTree using this file. Any methods or
 * secondary classes that you want are fine but we will only interact with those methods in the
 * DecisionTree framework.
 *
 * You must add code for the 1 member and 4 methods specified below.
 *
 * See DecisionTree for a description of default methods.
 */
public class DecisionTreeImpl extends DecisionTree {
  private DecTreeNode root;
  //ordered list of class labels
  private List<String> labels;
  //ordered list of attributes
  private List<String> attributes;
  //map to ordered discrete values taken by attributes
  private Map<String, List<String>> attributeValues;

  /**
   * Answers static questions about decision trees.
   */
  DecisionTreeImpl() {
    // no code necessary this is void purposefully
  }

  /**
   * Build a decision tree given only a training set.
   *
   * @param train: the training set
   */
  DecisionTreeImpl(DataSet train) {

    this.labels = train.labels;
    this.attributes = train.attributes;
    this.attributeValues = train.attributeValues;

    root = DecisionTreeLearning(train.instances ,this.attributes, null, null);
  }


  private String pluralityClass(List<Instance> instances){
    Map <String,Integer> att_count = new HashMap<>();
    Integer count;

    for(Instance instance : instances){
      count = att_count.get(instance.label);
      if (count == null)
        count = 0;
      att_count.put(instance.label, count+1);
    }

    int max=(Collections.max(att_count.values()));  // This will return max value in the Hashmap
    for (Map.Entry<String, Integer> entry : att_count.entrySet()) {
      if (entry.getValue()==max) {
        return entry.getKey();
      }
    }
    return null;
  }

  private boolean allSameClass(List<Instance> instances){
    String label = instances.get(0).label;
    for (Instance instance : instances) {
      if (!instance.label.equals(label)) {
        return false;
      }
    }
    return true;
  }


  private DecTreeNode DecisionTreeLearning(List<Instance> instances, List<String> attributes,  List<Instance> parents_instances, String parent_attr) {

    if (instances.size() == 0){
      String label = pluralityClass(parents_instances);
      return new DecTreeNode(label, null, parent_attr, true);
    }

    if (allSameClass(instances)){
      String label = instances.get(0).label;
      return new DecTreeNode(label, null, parent_attr, true);
    }

    if (attributes.size() == 0){
      String label = pluralityClass(instances);
      return new DecTreeNode(label, null, parent_attr, true);
    }

    String max_att_name = getMaxEntropyAtt(instances, attributes);
    DecTreeNode tree = new DecTreeNode(null, max_att_name, parent_attr, false);
    DecTreeNode subtree=null;
    List<String> max_att_values = this.attributeValues.get(max_att_name);
    Set<String> max_att_unique_vals = new HashSet<String>(max_att_values);
    List<Instance> atts_vals_examples;

    for (String max_att_unique_val: max_att_unique_vals){
      atts_vals_examples = getInstancesWithAttrVal(instances,attributes, max_att_name, max_att_unique_val);
      List<String> attributes_new = new ArrayList<String>(attributes);
      attributes_new.remove(getIndexByValue(attributes, max_att_name));
      subtree = DecisionTreeLearning(atts_vals_examples, attributes_new, instances, max_att_unique_val);
      tree.addChild(subtree);
    }
    return tree;
  }

  private List<Instance> getInstancesWithAttrVal(List<Instance> instances,List<String> attributes, String att_name, String att_val){
    List<Instance> atts_vals_examples = new ArrayList<>();
    int att_idx = getAttributeIndex(att_name);
    for (Instance inst: instances){
      if(inst.attributes.get(att_idx).equals(att_val))
        atts_vals_examples.add(inst);
    }
    return atts_vals_examples;
  }

  private String getMaxEntropyAtt(List<Instance> instances, List<String> attributes){
    double current_att_e;
    double max_att_e =-1;
    String max_att_name="";

    for (String att: attributes){
      current_att_e = entropyAttr(instances, att);
      if (current_att_e > max_att_e){
        max_att_e = current_att_e;
        max_att_name = att;
      }
    }
    return max_att_name;
  }

  @Override
  public String classify(Instance instance) {
    return classify(this.root, instance);
  }


  private String classify(DecTreeNode node, Instance instance){
    String label = null;
    if(node.terminal){
      return node.label;
    }
    int attr_idx = getAttributeIndex(node.attribute);
    for(DecTreeNode child : node.children){
      if(child.parentAttributeValue.equals(instance.attributes.get(attr_idx))){
        label = classify(child, instance);
      }
    }
    return label;
  }



  @Override
  public void rootInfoGain(DataSet train) {
    this.labels = train.labels;
    this.attributes = train.attributes;
    this.attributeValues = train.attributeValues;

    double root_entropy = entropyRoot(train.instances);
    double att_entropy;
    for (String att : attributes){
      att_entropy = entropyAttr(train.instances, att);
      System.out.print(att + " ");
      System.out.println(root_entropy-att_entropy);
    }

  }

  private double entropyRoot(List<Instance> instances){
    double sum = 0.0;
    int all_count = instances.size();
    Map <String,Integer> labels_count = new HashMap<>();
    Integer count;

    for(Instance instance : instances){
      count = labels_count.get(instance.label);
      if (count == null)
        count = 0;
      labels_count.put(instance.label, count+1);
    }

    for (Map.Entry<String, Integer> entry : labels_count.entrySet()) {
      double label_prob = (double)(entry.getValue())/(double)(all_count);
      sum += label_prob * Math.log(label_prob)/Math.log(2.0);
    }
    return -sum;
  }

  private double entropyAttr(List<Instance> instances, String attr) {
    // assumption - we have 2 classes (labels)
    double sum = 0.0;
    int all_count = instances.size();
    int att_index = getAttributeIndex(attr);
    Map <String,Integer> att_val_total_count = new HashMap<>();
    Map <String,Integer> att_labels_count = new HashMap<>();
    String label_to_count = instances.get(0).label;
    Integer count;

    for(Instance instance : instances){
      String instance_label = instance.label;
      if (instance_label.equals(label_to_count)){
        count = att_labels_count.get(instance.attributes.get(att_index));
        if (count == null)
          count = 0;
        att_labels_count.put(instance.attributes.get(att_index), count+1);
      }
      count = att_val_total_count.get(instance.attributes.get(att_index));
      if (count == null)
        count = 0;
      att_val_total_count.put(instance.attributes.get(att_index), count+1);
    }

    for (Map.Entry<String, Integer> entry : att_labels_count.entrySet()) {
      int att_val_total = att_val_total_count.get(entry.getKey());
      int att_label1_total = entry.getValue();
      double prob_val_lab= (double)(att_label1_total)/(double)(att_val_total);
      double att_prob= (double)(att_val_total)/(double)(all_count);
      if (prob_val_lab !=0 & prob_val_lab!=1){
        sum += att_prob * prob_val_lab*Math.log(prob_val_lab)/Math.log(2.0);
        sum += att_prob * (1-prob_val_lab)*Math.log(1-prob_val_lab)/Math.log(2.0);
      }

    }
    return -sum;
  }



  @Override
  public void printAccuracy(DataSet test) {
    int true_label = 0;
    int count=0;
    for (Instance instance : test.instances) {
      if(classify(instance).equals(test.instances.get(count).label)){
        true_label++;
      }
      count++;
    }
    double accuracy = (double)(true_label)/(double)(test.instances.size());
    System.out.println(accuracy);
  }

    /**
   * Build a decision tree given a training set then prune it using a tuning set.
   * ONLY for extra credits
   * @param train: the training set
   * @param tune: the tuning set
   */
  DecisionTreeImpl(DataSet train, DataSet tune) {

    this.labels = train.labels;
    this.attributes = train.attributes;
    this.attributeValues = train.attributeValues;
    // TODO: add code here
    // only for extra credits
  }

  @Override
  /**
   * Print the decision tree in the specified format
   */
  public void print() {

    printTreeNode(root, null, 0);
  }

  /**
   * Prints the subtree of the node with each line prefixed by 4 * k spaces.
   */
  public void printTreeNode(DecTreeNode p, DecTreeNode parent, int k) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < k; i++) {
      sb.append("    ");
    }
    String value;
    if (parent == null) {
      value = "ROOT";
    } else {
      int attributeValueIndex = this.getAttributeValueIndex(parent.attribute, p.parentAttributeValue);
      value = attributeValues.get(parent.attribute).get(attributeValueIndex);
    }
    sb.append(value);
    if (p.terminal) {
      sb.append(" (" + p.label + ")");
      System.out.println(sb.toString());
    } else {
      sb.append(" {" + p.attribute + "?}");
      System.out.println(sb.toString());
      for (DecTreeNode child : p.children) {
        printTreeNode(child, p, k + 1);
      }
    }
  }

  /**
   * Helper function to get the index of the label in labels list
   */
  private int getLabelIndex(String label) {
    for (int i = 0; i < this.labels.size(); i++) {
      if (label.equals(this.labels.get(i))) {
        return i;
      }
    }
    return -1;
  }

  /**
   * Helper function to get the index of the attribute in attributes list
   */
  private int getAttributeIndex(String attr) {
    for (int i = 0; i < this.attributes.size(); i++) {
      if (attr.equals(this.attributes.get(i))) {
        return i;
      }
    }
    return -1;
  }

  /**
   * Helper function to get the index of the attributeValue in the list for the attribute key in the attributeValues map
   */
  private int getAttributeValueIndex(String attr, String value) {
    for (int i = 0; i < attributeValues.get(attr).size(); i++) {
      if (value.equals(attributeValues.get(attr).get(i))) {
        return i;
      }
    }
    return -1;
  }

  private int getIndexByValue(List<String> list, String value) {
    for (int i = 0; i < list.size(); i++) {
      if (value.equals(list.get(i))) {
        return i;
      }
    }
    return -1;
  }

}
