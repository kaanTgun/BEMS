import java.util.ArrayList;
public interface TrieMapInterface{
  //Adds the key/value pair to the TrieMap
  public void put(String key, String value);
  
  //Returns the object value associated with the given key
  //If the key is not present in the TrieMap, returns null
  public String get(String key);
  
  //Returns true if key is in the TrieMap, false otherwise
  public boolean containsKey(String key);
  
  //Returns an ArrayList of objects containing all keys that start with prefix
  public ArrayList<String> getValuesForPrefix(String prefix);
  
  //Prints all values stored inside the TrieMap
  public void print();
  
}