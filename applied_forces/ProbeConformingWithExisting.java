import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ProbeConformingWithExisting {
  public static void main(String[] args) {
    Model m;
    try { m = ModelUtil.load("Model", "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph"); }
    catch (IOException e) { throw new RuntimeException(e); }

    try { m.component("comp1").mesh("mesh1").feature().remove("imptet"); } catch (Exception e) {}
    m.component("comp1").mesh("mesh1").feature().create("imptet", "Import");

    m.component("comp1").mesh("mesh1").feature("imptet").set("source", "nastran");
    m.component("comp1").mesh("mesh1").feature("imptet").set("filename", "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/__tracked_surface_tet_vol_conforming.bdf");
    try { m.component("comp1").mesh("mesh1").feature("imptet").set("domelem", "on"); } catch (Exception e) {}
    try { m.component("comp1").mesh("mesh1").feature("imptet").set("createdom", "on"); } catch (Exception e) {}
    try { m.component("comp1").mesh("mesh1").feature("imptet").set("linearelem", "on"); } catch (Exception e) {}

    try { System.out.println("source=" + m.component("comp1").mesh("mesh1").feature("imptet").getString("source")); } catch (Exception e) {}

    try {
      m.component("comp1").mesh("mesh1").run("imptet");
      System.out.println("imptet run ok");
    } catch (Exception e) {
      System.out.println("imptet run failed: " + e.getMessage());
      e.printStackTrace();
    }

    try {
      m.component("comp1").physics("solid").selection().all();
      int nd = m.component("comp1").physics("solid").selection().entities(3).length;
      int nb = m.component("comp1").physics("solid").selection().entities(2).length;
      int ne = m.component("comp1").physics("solid").selection().entities(1).length;
      System.out.println("counts dom=" + nd + " bnd=" + nb + " edge=" + ne);
    } catch (Exception e) {
      System.out.println("count read failed: " + e.getMessage());
    }

    try {
      m.component("comp1").physics("solid").feature("fix1").selection().geom("geom1",2);
      m.component("comp1").physics("solid").feature("fix1").selection().all();
    } catch (Exception e) {}

    try { m.component("comp1").physics("solid").feature().remove("body_probe"); } catch (Exception e) {}
    try {
      m.component("comp1").physics("solid").create("body_probe", "BodyLoad", 3);
      m.component("comp1").physics("solid").feature("body_probe").selection().all();
      m.component("comp1").physics("solid").feature("body_probe").set("FperVol", new String[]{"0","0","2e4[N/m^3]"});
    } catch (Exception e) {}

    try {
      m.study("std1").feature("stat").set("mesh", new String[][]{{"geom1","mesh1"}});
      m.study("std1").run();
      System.out.println("std1 run ok");
    } catch (Exception e) {
      System.out.println("std1 run failed: " + e.getMessage());
      e.printStackTrace();
    }

    try { m.result().numerical().remove("mxp"); } catch (Exception e) {}
    try {
      m.result().numerical().create("mxp", "MaxVolume");
      m.result().numerical("mxp").set("expr", new String[]{"solid.mises"});
      m.result().numerical("mxp").set("data", "dset6");
      m.result().numerical("mxp").setResult();
      double[][] r = m.result().numerical("mxp").getReal();
      System.out.println("max=" + (r!=null&&r.length>0&&r[0].length>0?r[0][0]:Double.NaN));
    } catch (Exception e) { System.out.println("max eval failed: " + e.getMessage()); }
  }
}
