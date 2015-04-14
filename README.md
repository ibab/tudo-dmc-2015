# Dortmund solution for DMC 2015

## Treffen am 14.04.2015

### Feature-Generation
* Macht euch Gedanke, so viel wie geht
* evtl. User Clustern

### Validierung
* Wir validieren mit dem Maß mit dem auch bewertet wird
* alle Schritte müssen validiert werden

### Abgabe

* Es ist besser bei `couponXUsed` {0, 1} anzugeben und nicht [0, 1].

### Datensatz

* `basePriceX` scheint soas wie die UVP zu sein / Was zur Hölle soll 0 heißen
* `priceX` ist dann sowas wie mit Rabatten, Steuern, Versand
* `rewardX` ist der Gewinn des Retailers 


### BasketValue

* Idee: Mean des BasketValue von vorherigen Bestellungen für bekannte User
* Clustering für unbekannte User, mean der Cluster
