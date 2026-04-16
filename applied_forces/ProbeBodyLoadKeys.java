import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;
import java.util.Arrays;

public class ProbeBodyLoadKeys {
  public static void main(String[] args) {
    Model m;
    try { m = ModelUtil.load("Model", "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph"); }
    catch (IOException e) { throw new RuntimeException(e); }

    try { m.component("comp1").physics("solid").feature().remove("body1"); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").create("body1", "BodyLoad", 3); System.out.println("BodyLoad create ok"); }
    catch (Exception e) { System.out.println("BodyLoad create failed: " + e.getMessage()); return; }

    String[] keys = new String[]{"FperVol","f","force","Fx","Fy","Fz","bodyforce","F"};
    for (String k: keys) {
      try {
        m.component("comp1").physics("solid").feature("body1").set(k, new String[]{"0","0","1"});
        System.out.println("set vec key " + k + " ok");
      } catch (Exception e) {
        try {
          m.component("comp1").physics("solid").feature("body1").set(k, "1");
          System.out.println("set scalar key " + k + " ok");
        } catch (Exception e2) {
          System.out.println("set key " + k + " failed");
        }
      }
    }

    try { m.save("/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/ProbeBodyLoadKeys_Model.mph"); }
    catch (IOException e) { throw new RuntimeException(e); }
  }
}
