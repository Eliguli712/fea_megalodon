import com.comsol.model.*;
import com.comsol.model.util.*;

public class CheckGreatWhiteJawMeshSource {
  private static final String MPH = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/custom_solver/great_white_jaw.mph";

  public static void main(String[] args) throws Exception {
    Model m = ModelUtil.load("Model", MPH);
    try {
      String src = m.component("comp1").mesh("mesh1").feature("imp1").getString("source");
      String fn = m.component("comp1").mesh("mesh1").feature("imp1").getString("filename");
      System.out.println("imp1 source=" + src);
      System.out.println("imp1 filename=" + fn);
    } catch (Exception e) {
      System.out.println("imp1 inspect failed: " + e.getMessage());
    }

    try {
      System.out.println("mesh tri=" + m.component("comp1").mesh("mesh1").getNumElem("tri"));
    } catch (Exception e) {
      System.out.println("mesh tri err=" + e.getMessage());
    }

    try {
      System.out.println("mesh tet=" + m.component("comp1").mesh("mesh1").getNumElem("tet"));
    } catch (Exception e) {
      System.out.println("mesh tet err=" + e.getMessage());
    }

    try {
      int dom = m.component("comp1").physics("solid").selection().entities(3).length;
      int bnd = m.component("comp1").physics("solid").selection().entities(2).length;
      System.out.println("solid selection dom=" + dom + " bnd=" + bnd);
    } catch (Exception e) {
      System.out.println("solid selection err=" + e.getMessage());
    }
  }
}
