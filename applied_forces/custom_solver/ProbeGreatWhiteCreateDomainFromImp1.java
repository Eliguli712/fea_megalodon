import com.comsol.model.*;
import com.comsol.model.util.*;

public class ProbeGreatWhiteCreateDomainFromImp1 {
  private static void p(String s){ System.out.println(s); }
  public static void main(String[] args) throws Exception {
    String in = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/custom_solver/great_white_jaw.mph";
    String out = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/custom_solver/ProbeGreatWhiteCreateDomainFromImp1_Model.mph";
    Model m = ModelUtil.load("Model", in);

    String[] keys = new String[]{"source","domelem","createdom","linearelem","facepartition","selectionstl","domelemsequence","meshtype","meshsize","detail","resdetail"};
    for(String k: keys){
      try {
        String[] a = m.component("comp1").mesh("mesh1").feature("imp1").getAllowedPropertyValues(k);
        if(a!=null) p("allowed " + k + "=" + java.util.Arrays.toString(a));
      } catch(Exception e){ p("allowed " + k + " failed: " + e.getMessage()); }
      try { p("get " + k + "=" + m.component("comp1").mesh("mesh1").feature("imp1").getString(k)); }
      catch(Exception e){ p("get " + k + " failed: " + e.getMessage()); }
    }

    String[][] sets = new String[][]{
      {"source","nastran"},
      {"domelem","on"},
      {"createdom","on"},
      {"linearelem","on"},
      {"facepartition","auto"},
      {"selectionstl","on"}
    };
    for(String[] kv: sets){
      try {
        m.component("comp1").mesh("mesh1").feature("imp1").set(kv[0], kv[1]);
        p("set " + kv[0] + "=" + kv[1] + " ok");
      } catch(Exception e){ p("set " + kv[0] + "=" + kv[1] + " failed: " + e.getMessage()); }
    }

    try {
      m.component("comp1").mesh("mesh1").run("imp1");
      p("run imp1 ok");
    } catch(Exception e){ p("run imp1 failed: " + e.getMessage()); }

    try {
      m.component("comp1").mesh("mesh1").run("fin");
      p("run fin ok");
    } catch(Exception e){ p("run fin failed: " + e.getMessage()); }

    try {
      int[] d = m.component("comp1").mesh("mesh1").feature("imp1").selection().entities(3);
      p("imp1 selection dim3 len=" + (d==null?-1:d.length));
    } catch(Exception e){ p("imp1 selection dim3 failed: " + e.getMessage()); }

    try {
      m.component("comp1").physics().create("solid", "SolidMechanics", "geom1");
    } catch(Exception e){}
    try {
      m.component("comp1").physics("solid").selection().all();
      int nd = m.component("comp1").physics("solid").selection().entities(3).length;
      int nb = m.component("comp1").physics("solid").selection().entities(2).length;
      p("solid selection counts dom=" + nd + " bnd=" + nb);
    } catch(Exception e){ p("solid selection check failed: " + e.getMessage()); }

    m.save(out);
    p("saved " + out);
  }
}
