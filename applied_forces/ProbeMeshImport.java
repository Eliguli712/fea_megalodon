import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ProbeMeshImport {
  public static void main(String[] args) {
    String in = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph";
    Model m;
    try { m = ModelUtil.load("Model", in); }
    catch (IOException e) { throw new RuntimeException(e); }

    try {
      try { m.component("comp1").mesh("mesh1").feature().remove("tmpimp"); } catch (Exception e) {}
      m.component("comp1").mesh("mesh1").feature().create("tmpimp", "Import");
      System.out.println("created comp1 mesh1 import feature");
      try {
        String[] a = m.component("comp1").mesh("mesh1").feature("tmpimp").getAllowedPropertyValues("source");
        System.out.println("allowed source values (comp1 mesh1 tmpimp):");
        if (a == null) System.out.println("  null");
        else for (String s : a) System.out.println("  " + s);
      } catch (Exception e) {
        System.out.println("could not get source allowed values comp1: " + e.getMessage());
      }

      String[] srcs = new String[]{"stl","mphbin","mphtxt","msh","nastran","unv","vtk","mesh"};
      for (String s : srcs) {
        try {
          m.component("comp1").mesh("mesh1").feature("tmpimp").set("source", s);
          String out = m.component("comp1").mesh("mesh1").feature("tmpimp").getString("source");
          System.out.println("set source="+s+" -> "+out);
        } catch (Exception e) {
          System.out.println("set source="+s+" failed: "+e.getMessage());
        }
      }
    } catch (Exception e) {
      System.out.println("comp1 mesh1 import create failed: " + e.getMessage());
    }

    try {
      String[] a = m.mesh("mpart1").feature("imp1").getAllowedPropertyValues("source");
      System.out.println("allowed source values (mesh mpart1 imp1):");
      if (a == null) System.out.println("  null");
      else for (String s : a) System.out.println("  " + s);
    } catch (Exception e) {
      System.out.println("mpart1 imp1 source allowed values failed: " + e.getMessage());
    }
  }
}
