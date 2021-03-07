# DecisionTreeLearning Algorithm Implementation :deciduous_tree:
### CS540 assignment

### Implemented functions:
1. **DecisionTreeImpl** - this is a recursive function, built according to figure 18.5, returns a decision tree.

    we have helper functions:
   
    ```boolean allSameClass(List<Instance> instances)```  -  Goes through all instance to check and returns true is all instances has the same label, else false.

    ```String pluralityClass(List<Instance> instances)``` - Goes through all instance and builds a hashmap saving each label's count. returns the labels with max count.

    ```double entropyRoot(List<Instance> instances)``` - Calculates entropy to root of the instances, returns the entropy.

    ```double entropyAttr(List<Instance> instances, String attr)```- Calculates the entropy to att (attribute name), returns the entropy.

    ```String getMaxIgAtt(List<Instance> instances, List<String> attributes)``` - First calculates entropy of the root using entropyRoot. 
   Then goes through all attributes, calculates entropy of all attributes using  entropyAttr. 
   Finally, goes through all attributes entropies and return the max IG attribute name.

    ```private List<Instance> getInstancesWithAttrVal(List<Instance> instances, String att_name, String att_val)```-
    This function goes through all instances, returns all instances which their attribute att_name = att_val

    DecisionTreeImpl itself-
    This recursive function has 3 stop conditions-
    * no more instances to go through, return a leaf with pluralityClass from the parents instances as a label.
    * all instances has the same label, return a leaf with this label
    * no more attributes left, return a leaf with pluralityClass from the instances as a label.
 
    if none of the above happened:
    * get max ig attribute using getMaxIgAtt, builds not terminal tree with that attribute.
    * builds a set of max ig attribute values
    * for each value, gets all instances with max ig attribute equals to the current value of the loop,
    and calling DecisionTreeImpl with this instances, the attributes minus the max ig attribute, and the current instances as parents.
    * finally, adding the tree builds as a child.

2. **classify**  - Recursive function that travels on the decision tree, according to the instance attributes values.
Stop condition is when arriving to a leaf. returns the label- this is the classification of the instance.

3. **rootInfoGain** - this method is based on the entropy and info gain equations. It uses entropyRoot and entropyAttr described above.
First, calculates root entropy using entropyRoot.
Then goes through all attributes, calculates for each one this entropy using entropyAttr and it's IG (entropyRoot-entropyAttr), printing it.

4. **printAccuracy** - this method prints the DTL accuracy.
For each instance, we check using classify what is the classification by the DTL, and what is the real classification.
we are printing the percentage of the instances we classified correctly.
