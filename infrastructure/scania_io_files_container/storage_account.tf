resource "azurerm_storage_account" "scania_storage_account" {
  name                     = var.scania_storage_account_name
  resource_group_name      = var.scania_resoure_group_name
  location                 = var.scania_resoure_group_location
  account_tier             = var.scania_account_tier
  account_replication_type = var.scania_account_replication_type
}