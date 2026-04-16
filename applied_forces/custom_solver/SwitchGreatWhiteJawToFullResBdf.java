import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class SwitchGreatWhiteJawToFullResBdf {
  private static final String MPH = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/custom_solver/great_white_jaw.mph";
  private static final String FULL_BDF = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/custom_solver/MBIE_White_Shark_HQ.bdf";

  public static void main(String[] args) throws Exception {
    Model m = ModelUtil.load("Model", MPH);

    try { m.component("comp1").mesh("mesh1").feature().remove("imp1"); } catch (Exception ignored) {}
    m.component("comp1").mesh("mesh1").feature().create("imp1", "Import");

    // Set filename first so COMSOL keeps Nastran mode for .bdf.
    m.component("comp1").mesh("mesh1").feature("imp1").set("filename", FULL_BDF);
    m.component("comp1").mesh("mesh1").feature("imp1").set("source", "nastran");

    System.out.println("imp1 source=" + m.component("comp1").mesh("mesh1").feature("imp1").getString("source"));
    System.out.println("imp1 filename=" + m.component("comp1").mesh("mesh1").feature("imp1").getString("filename"));

    m.component("comp1").mesh("mesh1").run("imp1");
    try { m.component("comp1").mesh("mesh1").run("fin"); } catch (Exception ignored) {}

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
      m.component("comp1").physics("solid").selection().all();
      int dom = m.component("comp1").physics("solid").selection().entities(3).length;
      int bnd = m.component("comp1").physics("solid").selection().entities(2).length;
      System.out.println("solid selection dom=" + dom + " bnd=" + bnd);
    } catch (Exception e) {
      System.out.println("solid selection err=" + e.getMessage());
    }

    try { m.save(MPH); }
    catch (IOException e) { throw new RuntimeException("save failed", e); }
    System.out.println("saved=" + MPH);
  }
}
