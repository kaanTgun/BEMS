//Note: All of your TrieMapInterface method implementations must function recursively
//I have left the method signatures from my own solution, which may be useful hints in how to approach the problem
//You are free to change/remove/etc. any of the methods, as long as your class still supports the TrieMapInterface
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

import javafx.print.Collation;

public class TrieMap implements TrieMapInterface{
  private TrieMapNode root;
  int i = 0;
  public TrieMap(){
    root = new TrieMapNode();

  }
  
  //Indirectly recursive method to meet definition of interface
  public void put(String key, String value){
    put(this.root, key, value);
  }
  
  //Recursive method
  //Note: arguments are only a suggestion, you may use your own if you devise a different recursive solution
  public void put(TrieMapNode current, String curKey, String value){
  
  
    if(curKey.length() > 1){
      Character newKey_C    = curKey.charAt(0);
      String    newKey_S    = curKey.substring(1, curKey.length());

      if (current.getChildren().get(newKey_C) != null ){
        put(current.getChildren().get(newKey_C), newKey_S, value);
      }
      else{
        TrieMapNode Node        = new TrieMapNode();
        current.getChildren().put(newKey_C, Node);
        put(current.getChildren().get(newKey_C), newKey_S, value);
      }
    }
    if (curKey.length()==1){
      if (current.getChildren().get(curKey.toCharArray()[0]) == null){
        TrieMapNode Node        = new TrieMapNode();
        Node.setValue(value);
        current.getChildren().put(curKey.toCharArray()[0], Node);
        current = this.root;
      }
      else{
        current.getChildren().get(curKey.toCharArray()[0]).setValue(value);
      }
    }
  
  }
  
  //Indirectly recursive method to meet definition of interface
  public String get(String key){
    String s = get(this.root, key, key);
    return s;
  }
  
  //Recursive method
  //Note: arguments are only a suggestion, you may use your own if you devise a different recursive solution
  public String get(TrieMapNode current, String curKey, String Val ){
    try{
      if(curKey.length()==1 && 
          current.getChildren().get(curKey.charAt(0)).getValue().equals(Val)){
        return current.getChildren().get(curKey.charAt(0)).getValue();
      }
      if (current.getChildren().get(curKey.charAt(0)) != null){
        return get(current.getChildren().get(curKey.charAt(0)),
                  curKey.substring(1, curKey.length()), 
                  Val);
      }
      return null;
    }
    catch (NullPointerException e) {
      return null;
    }
  }
  
  //Indirectly recursive method to meet definition of interface
  public boolean containsKey(String key){
    return containsKey(root, key, key);
  }
  
  //Recursive method
  //Note: arguments are only a suggestion, you may use your own if you devise a different recursive solution
  public boolean containsKey(TrieMapNode current, String curKey, String Val){
    try {
      if(curKey.length()==1){
        return current.getChildren().get(curKey.charAt(0)).getValue() == Val;
      }
      if (current.getChildren().get(curKey.charAt(0)) != null){
        return containsKey(
                current.getChildren().get(curKey.charAt(0)),
                curKey.substring(1, curKey.length()), 
                Val);
      }
      return false;
    }

    catch (NullPointerException e) {
      return false;
      
    }
  }
  public ArrayList<String> getValuesForPrefix(String prefix){
    ArrayList<String> Arr =  new  ArrayList<String>();
    TrieMapNode CurNode = findNode(root, prefix);
    return getValuesForPrefix(CurNode, prefix, Arr);
  }

  //Indirectly recursive method to meet definition of interface
  public ArrayList<String> getValuesForPrefix(TrieMapNode CurNode, String prefix, ArrayList<String> Arr ){
    if (CurNode == null){return Arr;}
    if (CurNode.getValue() !=null){
        Arr.add(CurNode.getValue());
    }
    if(CurNode.getChildren() != null){
      HashMap<Character, TrieMapNode> ChildNode = CurNode.getChildren();
      for(TrieMapNode N: ChildNode.values()){
        getValuesForPrefix(N, prefix, Arr);
      }
    }
    return Arr;
  }
  
  //Recursive helper function to find node that matches a prefix
  //Note: only a suggestion, you may solve the problem is any recursive manner
  public TrieMapNode findNode(TrieMapNode current, String curKey){
    if (curKey.length() == 0 && current.getChildren() != null){
      
      System.out.println(current);
      return current;
    }
    if (current.getChildren().get(curKey.charAt(0)) != null){
      
      TrieMapNode a = current.getChildren().get(curKey.charAt(0));
      String      s = curKey.substring(1, curKey.length());
      return findNode(a, s);
    }
    return null;
  }
  
  //Recursive helper function to get all keys in a node's subtree
  //Note: only a suggestion, you may solve the problem is any recursive manner
  public ArrayList<String> getKeys(TrieMapNode current){
    return new ArrayList<String>();
  }
  
  //Indirectly recursive method to meet definition of interface
  public void print(){
    print(root);
  }
  
  //Recursive method to print values in tree
  public void print(TrieMapNode current){
    if (current.getValue() != null){
      System.out.println(current.getValue());
    }
    if (current.getChildren() != null){
      HashMap<Character, TrieMapNode> ChildNode = current.getChildren();
      for(TrieMapNode N: ChildNode.values()){
        print(N);
      }
    }
    
  }
  
  public static void main(String[] args){
  }

}