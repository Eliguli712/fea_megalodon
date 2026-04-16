import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ProbeAutoAfterImport {
  private static void p(String s) { System.out.println(s); }

  public static void main(String[] args) {
    String in = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph";
    String out = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/ProbeAutoAfterImport_Model.mph";
    Model m;
    try { m = ModelUtil.load("Model", in); }
    catch (IOException e) { throw new RuntimeException(e); }

    try {
      try { m.component("comp1").mesh("mesh1").feature().remove("impmsh"); } catch (Exception e) {}
      m.component("comp1").mesh("mesh1").feature().create("impmsh", "Import");
      m.component("comp1").mesh("mesh1").feature("impmsh").set("source", "sequence");
      m.component("comp1").mesh("mesh1").feature("impmsh").set("sequence", "mpart1");
      m.component("comp1").mesh("mesh1").feature("impmsh").set("buildsource", "on");
      m.component("comp1").mesh("mesh1").feature("impmsh").set("domelemsequence", "on");
      m.component("comp1").mesh("mesh1").feature("impmsh").set("unmesheddom", "on");
      m.component("comp1").mesh("mesh1").run();
      p("mesh1 import run ok");
    } catch (Exception e) {
      p("mesh1 import run failed: " + e.getMessage());
      e.printStackTrace();
    }

    try {
      m.component("comp1").mesh("mesh1").automatic(true);
      p("mesh1 automatic(true) ok");
    } catch (Exception e) {
      p("mesh1 automatic(true) failed: " + e.getMessage());
    }

    try {
      m.component("comp1").mesh("mesh1").autoMeshSize(5);
      p("mesh1 autoMeshSize(5) ok");
    } catch (Exception e) {
      p("mesh1 autoMeshSize failed: " + e.getMessage());
    }

    try {
      m.component("comp1").mesh("mesh1").run();
      p("mesh1 automatic run ok");
    } catch (Exception e) {
      p("mesh1 automatic run failed: " + e.getMessage());
      e.printStackTrace();
    }

    try {
      m.save(out);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
}
