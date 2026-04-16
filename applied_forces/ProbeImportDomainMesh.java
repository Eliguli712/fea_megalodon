import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ProbeImportDomainMesh {
  private static void p(String s) { System.out.println(s); }
  private static void showAllowed(Model m, String key) {
    try {
      String[] a = m.component("comp1").mesh("mesh1").feature("tmpimp").getAllowedPropertyValues(key);
      if (a == null) {
        p("allowed " + key + " = null");
      } else {
        p("allowed " + key + ":");
        for (String v : a) p("  " + v);
      }
    } catch (Exception e) {
      p("allowed " + key + " failed: " + e.getMessage());
    }
  }

  private static void showGet(Model m, String key) {
    try {
      p("get " + key + " = " + m.component("comp1").mesh("mesh1").feature("tmpimp").getString(key));
    } catch (Exception e) {
      p("get " + key + " failed: " + e.getMessage());
    }
  }

  private static void trySet(Model m, String key, String val) {
    try {
      m.component("comp1").mesh("mesh1").feature("tmpimp").set(key, val);
      p("set " + key + "='" + val + "' ok -> " + m.component("comp1").mesh("mesh1").feature("tmpimp").getString(key));
    } catch (Exception e) {
      p("set " + key + "='" + val + "' failed: " + e.getMessage());
    }
  }

  public static void main(String[] args) {
    String in = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph";
    Model m;
    try { m = ModelUtil.load("Model", in); }
    catch (IOException e) { throw new RuntimeException(e); }

    try { m.component("comp1").mesh("mesh1").feature().remove("tmpimp"); } catch (Exception e) {}
    m.component("comp1").mesh("mesh1").feature().create("tmpimp", "Import");

    String[] keys = new String[] {
      "source", "domelem", "domelemsequence", "createdom", "modifiedcreatedom", "selectionstl",
      "meshtype", "meshsize", "resdetail", "detail", "narrowreg", "stltoltype", "stltolrel", "stltolabs"
    };

    for (String k : keys) {
      showAllowed(m, k);
      showGet(m, k);
    }

    trySet(m, "source", "sequence");
    trySet(m, "sequence", "mpart1");

    String[][] attempts = new String[][] {
      {"domelem", "on"},
      {"domelem", "off"},
      {"domelem", "all"},
      {"domelem", "free"},
      {"domelem", "tet"},
      {"domelemsequence", "on"},
      {"domelemsequence", "off"},
      {"createdom", "on"},
      {"createdom", "off"},
      {"modifiedcreatedom", "on"},
      {"modifiedcreatedom", "off"},
      {"meshtype", "tet"},
      {"meshtype", "tri"},
      {"meshtype", "auto"},
      {"meshsize", "custom"},
      {"meshsize", "normal"},
      {"resdetail", "1"},
      {"resdetail", "2"},
      {"resdetail", "3"},
      {"detail", "0.5"},
      {"narrowreg", "0.7"}
    };

    for (String[] kv : attempts) trySet(m, kv[0], kv[1]);

    try {
      m.component("comp1").mesh("mesh1").run();
      p("mesh1 run ok");
    } catch (Exception e) {
      p("mesh1 run failed: " + e.getMessage());
      e.printStackTrace();
    }

    try {
      m.save("/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/ProbeImportDomainMesh_Model.mph");
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
}
